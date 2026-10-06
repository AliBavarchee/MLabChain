// Mera / MLabChain C++ consensus & state core
// Reference implementation: Mera v0.2-devnet
// Copyright 2026 Ali Bavarchee
// SPDX-License-Identifier: Apache-2.0

#include <openssl/evp.h>
#include <openssl/sha.h>
#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

namespace {

using boost::multiprecision::uint128_t;

constexpr int PROTOCOL_VERSION = 2;
constexpr int ADDRESS_HEX_CHARS = 40;
constexpr uint64_t ATOMIC_PER_MERA = 100000000ULL;          // 8 decimals
constexpr uint64_t MAX_SUPPLY = 100000000ULL * ATOMIC_PER_MERA; // 100M MERA
constexpr uint64_t OPS_PER_MERA = 10000000ULL;             // reward scale
constexpr uint64_t MAX_WORK_REWARD = 50ULL * ATOMIC_PER_MERA;
constexpr uint64_t MIN_TX_FEE = 1ULL;                       // 1 atomic unit
constexpr uint64_t MAX_BLOCK_TX = 10000ULL;
constexpr int DEFAULT_DIFFICULTY = 4;

struct Tx {
    std::string txid;
    std::string type;
    std::string sender;
    std::string recipient;
    uint64_t amount = 0;
    uint64_t fee = 0;
    uint64_t nonce = 0;
    std::string challenge_id;
    std::string manifest_hash;
    std::string dataset_hash;
    std::string model_hash;
    std::string arch_hash;
    uint64_t ops = 0;
    uint64_t n_train = 0;
    uint64_t n_features = 0;
    uint64_t epochs = 0;
    uint64_t nmse_scaled = 0;
    uint64_t baseline_scaled = 0;
    uint64_t work_reward = 0;
    uint64_t wall_ms = 0;
    uint64_t lswu_micro = 0;
    std::string public_key;
    std::string signature;
    std::string payload_note;
};

struct Block {
    uint64_t height = 0;
    uint64_t timestamp_ms = 0;
    std::string previous_hash;
    std::string merkle_root;
    int difficulty = 0;
    uint64_t nonce = 0;
    std::string producer;
    std::string block_hash;
    uint64_t total_ops = 0;
    uint64_t total_rewards = 0;
};

class DB {
public:
    sqlite3* db = nullptr;
    explicit DB(const std::string& path) {
        if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
            std::string e = sqlite3_errmsg(db);
            sqlite3_close(db); db = nullptr;
            throw std::runtime_error("sqlite open failed: " + e);
        }
        exec("PRAGMA journal_mode=WAL;");
        exec("PRAGMA foreign_keys=ON;");
        exec("PRAGMA synchronous=FULL;");
    }
    ~DB() { if (db) sqlite3_close(db); }

    void exec(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string e = err ? err : "unknown sqlite error";
            sqlite3_free(err);
            throw std::runtime_error("sqlite: " + e);
        }
    }

    sqlite3_stmt* prepare(const std::string& sql) {
        sqlite3_stmt* st = nullptr;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK)
            throw std::runtime_error("sqlite prepare: " + std::string(sqlite3_errmsg(db)));
        return st;
    }

    static void bind(sqlite3_stmt* st, int idx, const std::string& s) {
        sqlite3_bind_text(st, idx, s.c_str(), -1, SQLITE_TRANSIENT);
    }
    static void bind(sqlite3_stmt* st, int idx, uint64_t v) {
        sqlite3_bind_int64(st, idx, static_cast<sqlite3_int64>(v));
    }
    static void bind(sqlite3_stmt* st, int idx, int v) { sqlite3_bind_int(st, idx, v); }

};

uint64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

bool devnet() {
    const char* n = std::getenv("MERA_NETWORK");
    return !n || std::string(n) == "devnet";
}

std::string network_name() {
    const char* n = std::getenv("MERA_NETWORK");
    return n ? n : "devnet";
}

std::string hex_bytes(const unsigned char* p, size_t n) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) out << std::setw(2) << static_cast<unsigned>(p[i]);
    return out.str();
}

std::vector<unsigned char> unhex(std::string s) {
    if (s.size() % 2) throw std::runtime_error("invalid hex length");
    std::vector<unsigned char> out(s.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        auto cvt = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int a = cvt(s[2*i]), b = cvt(s[2*i+1]);
        if (a < 0 || b < 0) throw std::runtime_error("invalid hex");
        out[i] = static_cast<unsigned char>((a << 4) | b);
    }
    return out;
}

std::string sha256(std::string_view data) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), digest);
    return hex_bytes(digest, SHA256_DIGEST_LENGTH);
}

std::string address_from_pubkey(const std::string& pub_hex) {
    auto p = unhex(pub_hex);
    if (p.size() != 32) throw std::runtime_error("Ed25519 public key must be 32 bytes");
    return "MERA1" + sha256(std::string_view(reinterpret_cast<const char*>(p.data()), p.size())).substr(0, ADDRESS_HEX_CHARS);
}

bool verify_ed25519(const std::string& pub_hex, const std::string& message, const std::string& sig_hex) {
    auto pub = unhex(pub_hex);
    auto sig = unhex(sig_hex);
    if (pub.size() != 32 || sig.size() != 64) return false;
    EVP_PKEY* key = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, pub.data(), pub.size());
    if (!key) return false;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    bool ok = false;
    if (ctx && EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, key) == 1) {
        ok = EVP_DigestVerify(ctx,
              sig.data(), sig.size(),
              reinterpret_cast<const unsigned char*>(message.data()), message.size()) == 1;
    }
    if (ctx) EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(key);
    return ok;
}

uint64_t u64(const std::string& s) {
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos, 10);
    if (pos != s.size()) throw std::runtime_error("invalid unsigned integer: " + s);
    return static_cast<uint64_t>(v);
}

std::string req(const std::map<std::string, std::string>& a, const std::string& k) {
    auto it = a.find(k);
    if (it == a.end()) throw std::runtime_error("missing argument --" + k);
    return it->second;
}

std::string opt(const std::map<std::string, std::string>& a, const std::string& k, const std::string& d = "") {
    auto it = a.find(k);
    return it == a.end() ? d : it->second;
}

std::map<std::string, std::string> args_map(int argc, char** argv, int start = 2) {
    std::map<std::string, std::string> a;
    for (int i = start; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) throw std::runtime_error("expected --key");
        k = k.substr(2);
        if (i + 1 >= argc || std::string(argv[i+1]).rfind("--", 0) == 0)
            a[k] = "true";
        else
            a[k] = argv[++i];
    }
    return a;
}

std::string transfer_message(const Tx& t) {
    return "TRANSFER|" + t.sender + "|" + t.recipient + "|" + std::to_string(t.amount) +
           "|" + std::to_string(t.fee) + "|" + std::to_string(t.nonce) + "|" + t.public_key;
}

std::string ml_message(const Tx& t) {
    return "ML_WORK|" + t.sender + "|" + t.challenge_id + "|" + t.manifest_hash + "|" +
           t.dataset_hash + "|" + t.model_hash + "|" + t.arch_hash + "|" + std::to_string(t.ops) +
           "|" + std::to_string(t.n_train) + "|" + std::to_string(t.n_features) + "|" +
           std::to_string(t.epochs) + "|" + std::to_string(t.nmse_scaled) + "|" +
           std::to_string(t.baseline_scaled) + "|" + std::to_string(t.work_reward) + "|" +
           std::to_string(t.wall_ms) + "|" + std::to_string(t.lswu_micro) + "|" +
           std::to_string(t.nonce) + "|" + t.public_key;
}

std::string txid(const Tx& t) {
    if (t.type == "TRANSFER") return sha256(transfer_message(t) + "|" + t.signature);
    if (t.type == "ML_WORK") return sha256(ml_message(t) + "|" + t.signature);
    if (t.type == "FAUCET") return sha256("FAUCET|" + t.recipient + "|" + std::to_string(t.amount));
    throw std::runtime_error("unknown tx type");
}

uint64_t quality_ppm(const Tx& t) {
    if (!t.baseline_scaled || t.nmse_scaled >= t.baseline_scaled) return 0;
    uint128_t num = uint128_t(t.baseline_scaled - t.nmse_scaled) * 1000000;
    uint64_t q = static_cast<uint64_t>(num / t.baseline_scaled);
    return std::min<uint64_t>(q, 1000000ULL);
}

uint64_t expected_work_reward(const Tx& t) {
    uint64_t q = quality_ppm(t);
    uint128_t num = uint128_t(t.ops) * q;
    uint64_t r = static_cast<uint64_t>((num * uint128_t(ATOMIC_PER_MERA)) / (uint128_t(OPS_PER_MERA) * 1000000));
    r = std::min<uint64_t>(r, MAX_WORK_REWARD);
    return r;
}

std::string merkle_root(const std::vector<std::string>& ids) {
    if (ids.empty()) return sha256("EMPTY");
    std::vector<std::string> h;
    h.reserve(ids.size());
    for (const auto& id : ids) h.push_back(sha256("L" + id));
    while (h.size() > 1) {
        if (h.size() % 2) h.push_back(h.back());
        std::vector<std::string> n;
        for (size_t i = 0; i < h.size(); i += 2) n.push_back(sha256("N" + h[i] + h[i+1]));
        h.swap(n);
    }
    return h[0];
}

std::string block_header(const Block& b) {
    return std::to_string(PROTOCOL_VERSION) + "|MERA|" + std::to_string(b.height) + "|" +
           std::to_string(b.timestamp_ms) + "|" + b.previous_hash + "|" + b.merkle_root + "|" +
           std::to_string(b.difficulty) + "|" + std::to_string(b.nonce) + "|" + b.producer + "|" +
           std::to_string(b.total_ops) + "|" + std::to_string(b.total_rewards);
}

std::string mine_hash(Block& b) {
    const std::string target(static_cast<size_t>(b.difficulty), '0');
    for (b.nonce = 0; ; ++b.nonce) {
        auto h = sha256(block_header(b));
        if (h.rfind(target, 0) == 0) return h;
        if (b.nonce == std::numeric_limits<uint64_t>::max())
            throw std::runtime_error("nonce exhausted");
    }
}

class State {
public:
    DB db;

    explicit State(const std::string& path) : db(path) { init_schema(); ensure_genesis(); }

    void init_schema() {
        db.exec("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);");
        db.exec("CREATE TABLE IF NOT EXISTS accounts (address TEXT PRIMARY KEY, balance INTEGER NOT NULL, nonce INTEGER NOT NULL);");
        db.exec("CREATE TABLE IF NOT EXISTS blocks (height INTEGER PRIMARY KEY, timestamp_ms INTEGER NOT NULL, previous_hash TEXT NOT NULL, merkle_root TEXT NOT NULL, difficulty INTEGER NOT NULL, nonce INTEGER NOT NULL, producer TEXT NOT NULL, block_hash TEXT NOT NULL UNIQUE, total_ops INTEGER NOT NULL, total_rewards INTEGER NOT NULL);");
        db.exec("CREATE TABLE IF NOT EXISTS txs (txid TEXT PRIMARY KEY, height INTEGER NOT NULL, ord INTEGER NOT NULL, type TEXT NOT NULL, sender TEXT NOT NULL, recipient TEXT NOT NULL, amount INTEGER NOT NULL, fee INTEGER NOT NULL, nonce INTEGER NOT NULL, challenge_id TEXT NOT NULL, manifest_hash TEXT NOT NULL, dataset_hash TEXT NOT NULL, model_hash TEXT NOT NULL, arch_hash TEXT NOT NULL, ops INTEGER NOT NULL, n_train INTEGER NOT NULL, n_features INTEGER NOT NULL, epochs INTEGER NOT NULL, nmse_scaled INTEGER NOT NULL, baseline_scaled INTEGER NOT NULL, work_reward INTEGER NOT NULL, wall_ms INTEGER NOT NULL, lswu_micro INTEGER NOT NULL, public_key TEXT NOT NULL, signature TEXT NOT NULL, payload_note TEXT NOT NULL);");
        db.exec("CREATE TABLE IF NOT EXISTS pending (seq INTEGER PRIMARY KEY AUTOINCREMENT, txid TEXT UNIQUE NOT NULL, type TEXT NOT NULL, sender TEXT NOT NULL, recipient TEXT NOT NULL, amount INTEGER NOT NULL, fee INTEGER NOT NULL, nonce INTEGER NOT NULL, challenge_id TEXT NOT NULL, manifest_hash TEXT NOT NULL, dataset_hash TEXT NOT NULL, model_hash TEXT NOT NULL, arch_hash TEXT NOT NULL, ops INTEGER NOT NULL, n_train INTEGER NOT NULL, n_features INTEGER NOT NULL, epochs INTEGER NOT NULL, nmse_scaled INTEGER NOT NULL, baseline_scaled INTEGER NOT NULL, work_reward INTEGER NOT NULL, wall_ms INTEGER NOT NULL, lswu_micro INTEGER NOT NULL, public_key TEXT NOT NULL, signature TEXT NOT NULL, payload_note TEXT NOT NULL);");
    }

    std::string meta(const std::string& k, const std::string& d = "") {
        auto st = db.prepare("SELECT v FROM meta WHERE k=?"); DB::bind(st,1,k);
        std::string v=d; if (sqlite3_step(st)==SQLITE_ROW) v=reinterpret_cast<const char*>(sqlite3_column_text(st,0)); sqlite3_finalize(st); return v;
    }
    void set_meta(const std::string& k, const std::string& v) {
        auto st=db.prepare("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v"); DB::bind(st,1,k); DB::bind(st,2,v); if(sqlite3_step(st)!=SQLITE_DONE){sqlite3_finalize(st);throw std::runtime_error(sqlite3_errmsg(db.db));} sqlite3_finalize(st);
    }

    void ensure_genesis() {
        if (!meta("network").empty()) return;
        set_meta("network", network_name());
        set_meta("protocol_version", std::to_string(PROTOCOL_VERSION));
        set_meta("asset", "MERA");
        set_meta("decimals", "8");
        set_meta("max_supply", std::to_string(MAX_SUPPLY));
        set_meta("ops_per_mera", std::to_string(OPS_PER_MERA));
        set_meta("genesis_hash", sha256("MERA-GENESIS-V2|" + network_name()));
        Block g; g.height=0; g.timestamp_ms=0; g.previous_hash=std::string(64,'0'); g.merkle_root=sha256("GENESIS"); g.difficulty=0; g.nonce=0; g.producer="GENESIS"; g.total_ops=0; g.total_rewards=0; g.block_hash=sha256(block_header(g));
        auto st=db.prepare("INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?,?,?)");
        DB::bind(st,1,g.height); DB::bind(st,2,g.timestamp_ms); DB::bind(st,3,g.previous_hash); DB::bind(st,4,g.merkle_root); DB::bind(st,5,g.difficulty); DB::bind(st,6,g.nonce); DB::bind(st,7,g.producer); DB::bind(st,8,g.block_hash); DB::bind(st,9,g.total_ops); DB::bind(st,10,g.total_rewards);
        if(sqlite3_step(st)!=SQLITE_DONE){sqlite3_finalize(st);throw std::runtime_error(sqlite3_errmsg(db.db));} sqlite3_finalize(st);
    }

    uint64_t balance(const std::string& address) {
        auto st=db.prepare("SELECT balance FROM accounts WHERE address=?"); DB::bind(st,1,address); uint64_t b=0; if(sqlite3_step(st)==SQLITE_ROW) b=static_cast<uint64_t>(sqlite3_column_int64(st,0)); sqlite3_finalize(st); return b;
    }
    uint64_t nonce(const std::string& address) {
        auto st=db.prepare("SELECT nonce FROM accounts WHERE address=?"); DB::bind(st,1,address); uint64_t n=0; if(sqlite3_step(st)==SQLITE_ROW) n=static_cast<uint64_t>(sqlite3_column_int64(st,0)); sqlite3_finalize(st); return n;
    }
    void set_account(const std::string& address, uint64_t bal, uint64_t n) {
        auto st=db.prepare("INSERT INTO accounts(address,balance,nonce) VALUES(?,?,?) ON CONFLICT(address) DO UPDATE SET balance=excluded.balance,nonce=excluded.nonce"); DB::bind(st,1,address); DB::bind(st,2,bal); DB::bind(st,3,n); if(sqlite3_step(st)!=SQLITE_DONE){sqlite3_finalize(st);throw std::runtime_error(sqlite3_errmsg(db.db));} sqlite3_finalize(st);
    }
    uint64_t total_supply() {
        auto st=db.prepare("SELECT COALESCE(SUM(balance),0) FROM accounts"); uint64_t x=0; if(sqlite3_step(st)==SQLITE_ROW)x=static_cast<uint64_t>(sqlite3_column_int64(st,0)); sqlite3_finalize(st); return x;
    }

    uint64_t height() {
        auto st=db.prepare("SELECT MAX(height) FROM blocks"); uint64_t h=0; if(sqlite3_step(st)==SQLITE_ROW) h=static_cast<uint64_t>(sqlite3_column_int64(st,0)); sqlite3_finalize(st); return h;
    }
    std::string tip_hash() {
        auto st=db.prepare("SELECT block_hash FROM blocks ORDER BY height DESC LIMIT 1"); std::string h; if(sqlite3_step(st)==SQLITE_ROW)h=reinterpret_cast<const char*>(sqlite3_column_text(st,0)); sqlite3_finalize(st); return h;
    }

    void validate_signed_common(const Tx& t, const std::string& message) {
        if (t.sender != address_from_pubkey(t.public_key)) throw std::runtime_error("sender does not match public key");
        if (!verify_ed25519(t.public_key, message, t.signature)) throw std::runtime_error("invalid Ed25519 signature");
        uint64_t n = nonce(t.sender);
        if (t.nonce != n + 1) throw std::runtime_error("invalid account nonce");
    }

    void validate_tx(const Tx& t) {
        if (t.type == "TRANSFER") {
            if (t.recipient.rfind("MERA1",0)!=0) throw std::runtime_error("invalid recipient address");
            if (t.amount==0) throw std::runtime_error("transfer amount must be > 0");
            if (t.fee < MIN_TX_FEE) throw std::runtime_error("fee below minimum");
            validate_signed_common(t, transfer_message(t));
            uint128_t need=uint128_t(t.amount)+t.fee;
            if(uint128_t(balance(t.sender))<need) throw std::runtime_error("insufficient balance");
            return;
        }
        if (t.type == "ML_WORK") {
            if (t.challenge_id.empty() || t.manifest_hash.size()!=64 || t.dataset_hash.size()!=64 || t.model_hash.size()!=64 || t.arch_hash.size()!=64)
                throw std::runtime_error("incomplete ML challenge/artifact hashes");
            if (!t.ops || !t.n_train || !t.n_features || !t.epochs) throw std::runtime_error("ML work fields must be nonzero");
            if (!t.baseline_scaled || !t.nmse_scaled) throw std::runtime_error("NMSE fields must be positive fixed-point integers");
            if (expected_work_reward(t) != t.work_reward) throw std::runtime_error("work reward formula mismatch");
            if (t.work_reward==0) throw std::runtime_error("model does not improve on pinned baseline; no Mera reward");
            validate_signed_common(t, ml_message(t));
            if(uint128_t(total_supply())+t.work_reward>MAX_SUPPLY) throw std::runtime_error("maximum supply exceeded");
            return;
        }
        if (t.type == "FAUCET") {
            if (meta("network") != "devnet") throw std::runtime_error("faucet is disabled outside devnet");
            if (t.amount==0 || t.recipient.rfind("MERA1",0)!=0) throw std::runtime_error("invalid faucet transaction");
            if(uint128_t(total_supply())+t.amount>MAX_SUPPLY) throw std::runtime_error("maximum supply exceeded");
            return;
        }
        throw std::runtime_error("unsupported transaction type");
    }

    void apply_tx(const Tx& t) {
        if (t.type == "TRANSFER") {
            uint64_t sb=balance(t.sender); uint64_t rb=balance(t.recipient);
            set_account(t.sender,sb-t.amount-t.fee,nonce(t.sender)+1);
            set_account(t.recipient,rb+t.amount,nonce(t.recipient));
        } else if (t.type == "ML_WORK") {
            uint64_t b=balance(t.sender);
            set_account(t.sender,b+t.work_reward,nonce(t.sender)+1);
        } else if (t.type == "FAUCET") {
            set_account(t.recipient,balance(t.recipient)+t.amount,nonce(t.recipient));
        }
    }

    void insert_pending(const Tx& t) {
        auto st=db.prepare("INSERT INTO pending(txid,type,sender,recipient,amount,fee,nonce,challenge_id,manifest_hash,dataset_hash,model_hash,arch_hash,ops,n_train,n_features,epochs,nmse_scaled,baseline_scaled,work_reward,wall_ms,lswu_micro,public_key,signature,payload_note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
        int i=1; auto bs=[&](const std::string&s){DB::bind(st,i++,s);}; auto bu=[&](uint64_t v){DB::bind(st,i++,v);};
        bs(t.txid);bs(t.type);bs(t.sender);bs(t.recipient);bu(t.amount);bu(t.fee);bu(t.nonce);bs(t.challenge_id);bs(t.manifest_hash);bs(t.dataset_hash);bs(t.model_hash);bs(t.arch_hash);bu(t.ops);bu(t.n_train);bu(t.n_features);bu(t.epochs);bu(t.nmse_scaled);bu(t.baseline_scaled);bu(t.work_reward);bu(t.wall_ms);bu(t.lswu_micro);bs(t.public_key);bs(t.signature);bs(t.payload_note);
        if(sqlite3_step(st)!=SQLITE_DONE){std::string e=sqlite3_errmsg(db.db);sqlite3_finalize(st);throw std::runtime_error("pending insert: "+e);} sqlite3_finalize(st);
    }

    std::vector<Tx> pending_all() {
        std::vector<Tx> out; auto st=db.prepare("SELECT txid,type,sender,recipient,amount,fee,nonce,challenge_id,manifest_hash,dataset_hash,model_hash,arch_hash,ops,n_train,n_features,epochs,nmse_scaled,baseline_scaled,work_reward,wall_ms,lswu_micro,public_key,signature,payload_note FROM pending ORDER BY seq");
        while(sqlite3_step(st)==SQLITE_ROW) { out.push_back(row_tx(st,0)); }
        sqlite3_finalize(st);
        return out;
    }

    static std::string coltext(sqlite3_stmt* st,int c){ const unsigned char* p=sqlite3_column_text(st,c); return p?reinterpret_cast<const char*>(p):""; }
    static Tx row_tx(sqlite3_stmt* st,int off) {
        Tx t; int c=off; t.txid=coltext(st,c++); t.type=coltext(st,c++); t.sender=coltext(st,c++); t.recipient=coltext(st,c++); t.amount=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.fee=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.nonce=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.challenge_id=coltext(st,c++); t.manifest_hash=coltext(st,c++); t.dataset_hash=coltext(st,c++); t.model_hash=coltext(st,c++); t.arch_hash=coltext(st,c++); t.ops=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.n_train=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.n_features=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.epochs=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.nmse_scaled=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.baseline_scaled=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.work_reward=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.wall_ms=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.lswu_micro=static_cast<uint64_t>(sqlite3_column_int64(st,c++)); t.public_key=coltext(st,c++); t.signature=coltext(st,c++); t.payload_note=coltext(st,c++); return t;
    }

    void submit(Tx t) {
        if (t.type=="TRANSFER") t.signature = t.signature;
        t.txid=txid(t);
        validate_tx(t);
        auto st=db.prepare("SELECT 1 FROM txs WHERE txid=? UNION SELECT 1 FROM pending WHERE txid=? LIMIT 1"); DB::bind(st,1,t.txid); DB::bind(st,2,t.txid); bool exists=sqlite3_step(st)==SQLITE_ROW; sqlite3_finalize(st); if(exists) throw std::runtime_error("duplicate transaction");
        insert_pending(t);
        std::cout << "TXID=" << t.txid << "\nSTATUS=accepted\n";
    }

    void submit_faucet(const std::string& recipient,uint64_t amount){ if(meta("network")!="devnet") throw std::runtime_error("faucet is disabled outside devnet"); Tx t; t.type="FAUCET"; t.sender="SYSTEM"; t.recipient=recipient; t.amount=amount; t.txid=txid(t); validate_tx(t); insert_pending(t); std::cout<<"TXID="<<t.txid<<"\nSTATUS=accepted\n"; }

    void mine(int difficulty,const std::string& producer) {
        auto p=pending_all(); if(p.empty()) throw std::runtime_error("no pending transactions"); if(p.size()>MAX_BLOCK_TX) throw std::runtime_error("pending transaction count exceeds block limit");
        db.exec("BEGIN IMMEDIATE;");
        try {
            // Transaction validation + state transitions are performed inside the same DB transaction.
            uint64_t ops=0,rewards=0;
            for(auto &t:p){ validate_tx(t); apply_tx(t); ops += t.ops; rewards += t.work_reward; }
            Block b; b.height=height()+1; b.timestamp_ms=now_ms(); b.previous_hash=tip_hash(); std::vector<std::string> ids; for(auto&t:p)ids.push_back(t.txid); b.merkle_root=merkle_root(ids); b.difficulty=difficulty; b.producer=producer; b.total_ops=ops; b.total_rewards=rewards; auto start=std::chrono::steady_clock::now(); b.block_hash=mine_hash(b); auto end=std::chrono::steady_clock::now();
            auto st=db.prepare("INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?,?,?)"); int i=1; DB::bind(st,i++,b.height);DB::bind(st,i++,b.timestamp_ms);DB::bind(st,i++,b.previous_hash);DB::bind(st,i++,b.merkle_root);DB::bind(st,i++,b.difficulty);DB::bind(st,i++,b.nonce);DB::bind(st,i++,b.producer);DB::bind(st,i++,b.block_hash);DB::bind(st,i++,b.total_ops);DB::bind(st,i++,b.total_rewards); if(sqlite3_step(st)!=SQLITE_DONE){std::string e=sqlite3_errmsg(db.db);sqlite3_finalize(st);throw std::runtime_error(e);} sqlite3_finalize(st);
            for(size_t ord=0;ord<p.size();++ord){auto&t=p[ord]; auto q=db.prepare("INSERT INTO txs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"); int c=1; auto bs=[&](const std::string&s){DB::bind(q,c++,s);}; auto bu=[&](uint64_t v){DB::bind(q,c++,v);}; bs(t.txid);bu(b.height);bu(ord);bs(t.type);bs(t.sender);bs(t.recipient);bu(t.amount);bu(t.fee);bu(t.nonce);bs(t.challenge_id);bs(t.manifest_hash);bs(t.dataset_hash);bs(t.model_hash);bs(t.arch_hash);bu(t.ops);bu(t.n_train);bu(t.n_features);bu(t.epochs);bu(t.nmse_scaled);bu(t.baseline_scaled);bu(t.work_reward);bu(t.wall_ms);bu(t.lswu_micro);bs(t.public_key);bs(t.signature);bs(t.payload_note); if(sqlite3_step(q)!=SQLITE_DONE){std::string e=sqlite3_errmsg(db.db);sqlite3_finalize(q);throw std::runtime_error(e);} sqlite3_finalize(q);}
            db.exec("DELETE FROM pending;");
            db.exec("COMMIT;");
            double sec=std::chrono::duration<double>(end-start).count();
            std::cout<<"HEIGHT="<<b.height<<"\nBLOCK_HASH="<<b.block_hash<<"\nTOTAL_OPS="<<b.total_ops<<"\nTOTAL_REWARD_ATOMIC="<<b.total_rewards<<"\nPOW_SECONDS="<<std::fixed<<std::setprecision(6)<<sec<<"\n";
        } catch (...) { db.exec("ROLLBACK;"); throw; }
    }

    void validate_chain() {
        std::cout << "NETWORK=" << network_name() << "\n";
        auto st=db.prepare("SELECT height,timestamp_ms,previous_hash,merkle_root,difficulty,nonce,producer,block_hash,total_ops,total_rewards FROM blocks ORDER BY height");
        std::string prev(64,'0'); uint64_t last_h=0; bool first=true; size_t blocks=0,txs=0;
        std::map<std::string, std::pair<uint64_t,uint64_t>> replay; // address -> {balance, nonce}
        uint64_t replay_supply=0;
        while(sqlite3_step(st)==SQLITE_ROW){
            Block b; int c=0; b.height=static_cast<uint64_t>(sqlite3_column_int64(st,c++));
            b.timestamp_ms=static_cast<uint64_t>(sqlite3_column_int64(st,c++));
            b.previous_hash=reinterpret_cast<const char*>(sqlite3_column_text(st,c++));
            b.merkle_root=reinterpret_cast<const char*>(sqlite3_column_text(st,c++));
            b.difficulty=sqlite3_column_int(st,c++); b.nonce=static_cast<uint64_t>(sqlite3_column_int64(st,c++));
            b.producer=reinterpret_cast<const char*>(sqlite3_column_text(st,c++));
            b.block_hash=reinterpret_cast<const char*>(sqlite3_column_text(st,c++));
            b.total_ops=static_cast<uint64_t>(sqlite3_column_int64(st,c++));
            b.total_rewards=static_cast<uint64_t>(sqlite3_column_int64(st,c++));
            if(first){
                if(b.height!=0 || b.previous_hash!=std::string(64,'0')){sqlite3_finalize(st);throw std::runtime_error("invalid genesis block");}
            } else if(b.height!=last_h+1 || b.previous_hash!=prev){sqlite3_finalize(st);throw std::runtime_error("broken block chain");}
            if(b.block_hash!=sha256(block_header(b))){sqlite3_finalize(st);throw std::runtime_error("block hash mismatch at height "+std::to_string(b.height));}
            if(b.difficulty>0 && b.block_hash.rfind(std::string(b.difficulty,'0'),0)!=0){sqlite3_finalize(st);throw std::runtime_error("PoW invalid at height "+std::to_string(b.height));}

            std::vector<std::string> ids;
            auto q=db.prepare("SELECT txid FROM txs WHERE height=? ORDER BY ord"); DB::bind(q,1,b.height);
            while(sqlite3_step(q)==SQLITE_ROW) ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(q,0)));
            sqlite3_finalize(q);
            if(b.height>0 && b.merkle_root!=merkle_root(ids)){sqlite3_finalize(st);throw std::runtime_error("Merkle root mismatch at height "+std::to_string(b.height));}

            uint64_t bo=0, br=0;
            auto t=db.prepare("SELECT txid,type,sender,recipient,amount,fee,nonce,challenge_id,manifest_hash,dataset_hash,model_hash,arch_hash,ops,n_train,n_features,epochs,nmse_scaled,baseline_scaled,work_reward,wall_ms,lswu_micro,public_key,signature,payload_note FROM txs WHERE height=? ORDER BY ord");
            DB::bind(t,1,b.height);
            while(sqlite3_step(t)==SQLITE_ROW){
                Tx x=row_tx(t,0);
                if(x.txid!=txid(x)){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("txid mismatch at block "+std::to_string(b.height));}
                validate_signature_for_stored(x);
                auto &acct=replay[x.sender];
                if(x.type=="TRANSFER"){
                    if(x.nonce!=acct.second+1){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("nonce mismatch for "+x.sender);}
                    uint128_t need=uint128_t(x.amount)+x.fee;
                    if(uint128_t(acct.first)<need){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("negative balance in replay for "+x.sender);}
                    acct.first-=x.amount+x.fee; acct.second=x.nonce;
                    replay[x.recipient].first+=x.amount;
                    replay_supply-=x.fee; // fees are burned
                } else if(x.type=="ML_WORK"){
                    if(x.nonce!=acct.second+1){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("ML nonce mismatch for "+x.sender);}
                    if(expected_work_reward(x)!=x.work_reward || !x.work_reward){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("invalid ML reward for "+x.txid);}
                    if(uint128_t(replay_supply)+x.work_reward>MAX_SUPPLY){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("supply cap exceeded during replay");}
                    acct.first+=x.work_reward; acct.second=x.nonce; replay_supply+=x.work_reward;
                } else if(x.type=="FAUCET"){
                    if(meta("network")!="devnet"){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("faucet tx exists outside devnet");}
                    if(uint128_t(replay_supply)+x.amount>MAX_SUPPLY){sqlite3_finalize(t);sqlite3_finalize(st);throw std::runtime_error("supply cap exceeded during faucet replay");}
                    replay[x.recipient].first+=x.amount; replay_supply+=x.amount;
                }
                bo+=x.ops; br+=x.work_reward; txs++;
            }
            sqlite3_finalize(t);
            if(bo!=b.total_ops||br!=b.total_rewards){sqlite3_finalize(st);throw std::runtime_error("block counters mismatch at height "+std::to_string(b.height));}
            prev=b.block_hash; last_h=b.height; first=false; blocks++;
        }
        sqlite3_finalize(st);
        uint64_t final_supply=total_supply();
        if(replay_supply!=final_supply){throw std::runtime_error("account-state supply mismatch: replay="+std::to_string(replay_supply)+" db="+std::to_string(final_supply));}
        if(final_supply>MAX_SUPPLY)throw std::runtime_error("supply cap exceeded");
        std::cout<<"STATUS=VALID\nBLOCKS="<<blocks<<"\nTRANSACTIONS="<<txs<<"\nTOTAL_SUPPLY_ATOMIC="<<final_supply<<"\nTOTAL_SUPPLY_MERA="<<std::fixed<<std::setprecision(8)<<(double)final_supply/ATOMIC_PER_MERA<<"\n";
    }

    void validate_signature_for_stored(const Tx& t) {
        if(t.type=="TRANSFER") validate_tx_signature_only(t,transfer_message(t));
        else if(t.type=="ML_WORK") validate_tx_signature_only(t,ml_message(t));
        else if(t.type=="FAUCET") { if(!devnet()) throw std::runtime_error("faucet tx on non-devnet"); }
        else throw std::runtime_error("unknown stored tx type");
    }
    void validate_tx_signature_only(const Tx& t,const std::string&m){ if(t.sender!=address_from_pubkey(t.public_key)||!verify_ed25519(t.public_key,m,t.signature))throw std::runtime_error("signature invalid for "+t.txid); }

    void show_status(){ uint64_t h=height(); auto b=total_supply(); auto st=db.prepare("SELECT COUNT(*) FROM pending");uint64_t p=0;if(sqlite3_step(st)==SQLITE_ROW)p=sqlite3_column_int64(st,0);sqlite3_finalize(st);auto tx=db.prepare("SELECT COUNT(*) FROM txs");uint64_t n=0;if(sqlite3_step(tx)==SQLITE_ROW)n=sqlite3_column_int64(tx,0);sqlite3_finalize(tx);auto ops=db.prepare("SELECT COALESCE(SUM(ops),0),COALESCE(SUM(work_reward),0) FROM txs");uint64_t o=0,r=0;if(sqlite3_step(ops)==SQLITE_ROW){o=sqlite3_column_int64(ops,0);r=sqlite3_column_int64(ops,1);}sqlite3_finalize(ops);std::cout<<"NETWORK="<<network_name()<<"\nHEIGHT="<<h<<"\nTIP="<<tip_hash()<<"\nPENDING="<<p<<"\nTRANSACTIONS="<<n<<"\nTOTAL_OPS="<<o<<"\nTOTAL_MERA_REWARDS="<<std::fixed<<std::setprecision(8)<<(double)r/ATOMIC_PER_MERA<<"\nTOTAL_SUPPLY_MERA="<<std::fixed<<std::setprecision(8)<<(double)b/ATOMIC_PER_MERA<<"\nMAX_SUPPLY_MERA=100000000.00000000\n"; }

    Tx load_tx(const std::string&id){auto st=db.prepare("SELECT txid,type,sender,recipient,amount,fee,nonce,challenge_id,manifest_hash,dataset_hash,model_hash,arch_hash,ops,n_train,n_features,epochs,nmse_scaled,baseline_scaled,work_reward,wall_ms,lswu_micro,public_key,signature,payload_note FROM txs WHERE txid=?");DB::bind(st,1,id);if(sqlite3_step(st)!=SQLITE_ROW){sqlite3_finalize(st);throw std::runtime_error("transaction not found");}Tx t=row_tx(st,0);sqlite3_finalize(st);return t;}

    void print_tx(const Tx&t){std::cout<<"TXID="<<t.txid<<"\nTYPE="<<t.type<<"\nSENDER="<<t.sender<<"\nRECIPIENT="<<t.recipient<<"\nAMOUNT_ATOMIC="<<t.amount<<"\nFEE_ATOMIC="<<t.fee<<"\nNONCE="<<t.nonce<<"\nCHALLENGE_ID="<<t.challenge_id<<"\nMANIFEST_SHA256="<<t.manifest_hash<<"\nDATASET_SHA256="<<t.dataset_hash<<"\nMODEL_SHA256="<<t.model_hash<<"\nARCH_SHA256="<<t.arch_hash<<"\nOPS="<<t.ops<<"\nN_TRAIN="<<t.n_train<<"\nN_FEATURES="<<t.n_features<<"\nEPOCHS="<<t.epochs<<"\nNMSE_SCALED="<<t.nmse_scaled<<"\nBASELINE_SCALED="<<t.baseline_scaled<<"\nWORK_REWARD_ATOMIC="<<t.work_reward<<"\nWALL_MS="<<t.wall_ms<<"\nLSWU_MICRO="<<t.lswu_micro<<"\nPUBLIC_KEY="<<t.public_key<<"\nSIGNATURE="<<t.signature<<"\nNOTE="<<t.payload_note<<"\n";}
};

void usage(){
    std::cout << R"TXT(Mera core v0.2\nCommands:\n  init\n  balance --address MERA1...\n  nonce --address MERA1...\n  submit-transfer --sender --recipient --amount --fee --nonce --public-key --signature\n  submit-ml --sender --challenge-id --manifest-sha256 --dataset-sha256 --model-sha256 --arch-sha256 --ops --n-train --n-features --epochs --nmse-scaled --baseline-scaled --work-reward --wall-ms --lswu-micro --nonce --public-key --signature [--note]\n  faucet --recipient --amount      (devnet only)\n  mine --difficulty N --producer MERA1...\n  show-tx --txid HEX\n  status\n  validate\n)TXT";
}

} // namespace

int main(int argc,char**argv){
    try{
        if(argc<2){usage();return 0;}
        std::string cmd=argv[1];
        std::string path="mera_data/mera.sqlite";
        std::filesystem::create_directories("mera_data");
        State s(path);
        if(cmd=="init"){std::cout<<"STATUS=initialized\nNETWORK="<<network_name()<<"\nGENESIS="<<s.meta("genesis_hash")<<"\nASSET=MERA\nDECIMALS=8\nMAX_SUPPLY=100000000\n";return 0;}
        if(cmd=="balance"){auto a=args_map(argc,argv);auto b=s.balance(req(a,"address"));std::cout<<"BALANCE_ATOMIC="<<b<<"\nBALANCE_MERA="<<std::fixed<<std::setprecision(8)<<(double)b/ATOMIC_PER_MERA<<"\n";return 0;}
        if(cmd=="nonce"){auto a=args_map(argc,argv);std::cout<<"NONCE="<<s.nonce(req(a,"address"))<<"\n";return 0;}
        if(cmd=="submit-transfer"){auto a=args_map(argc,argv);Tx t;t.type="TRANSFER";t.sender=req(a,"sender");t.recipient=req(a,"recipient");t.amount=u64(req(a,"amount"));t.fee=u64(req(a,"fee"));t.nonce=u64(req(a,"nonce"));t.public_key=req(a,"public-key");t.signature=req(a,"signature");t.payload_note=opt(a,"note");s.submit(t);return 0;}
        if(cmd=="submit-ml"){auto a=args_map(argc,argv);Tx t;t.type="ML_WORK";t.sender=req(a,"sender");t.challenge_id=req(a,"challenge-id");t.manifest_hash=req(a,"manifest-sha256");t.dataset_hash=req(a,"dataset-sha256");t.model_hash=req(a,"model-sha256");t.arch_hash=req(a,"arch-sha256");t.ops=u64(req(a,"ops"));t.n_train=u64(req(a,"n-train"));t.n_features=u64(req(a,"n-features"));t.epochs=u64(req(a,"epochs"));t.nmse_scaled=u64(req(a,"nmse-scaled"));t.baseline_scaled=u64(req(a,"baseline-scaled"));t.work_reward=u64(req(a,"work-reward"));t.wall_ms=u64(req(a,"wall-ms"));t.lswu_micro=u64(req(a,"lswu-micro"));t.nonce=u64(req(a,"nonce"));t.public_key=req(a,"public-key");t.signature=req(a,"signature");t.payload_note=opt(a,"note");s.submit(t);return 0;}
        if(cmd=="faucet"){auto a=args_map(argc,argv);s.submit_faucet(req(a,"recipient"),u64(req(a,"amount")));return 0;}
        if(cmd=="mine"){auto a=args_map(argc,argv);int d=std::stoi(opt(a,"difficulty",std::to_string(DEFAULT_DIFFICULTY)));auto p=opt(a,"producer","SYSTEM");s.mine(d,p);return 0;}
        if(cmd=="show-tx"){auto a=args_map(argc,argv);s.print_tx(s.load_tx(req(a,"txid")));return 0;}
        if(cmd=="status"){s.show_status();return 0;}
        if(cmd=="validate"){s.validate_chain();return 0;}
        usage(); return 1;
    }catch(const std::exception&e){std::cerr<<"ERROR="<<e.what()<<"\n";return 2;}
}
