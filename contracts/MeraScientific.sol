// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Capped} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title Mera Scientific Work Token
/// @notice Exchange-compatible ERC-20 settlement representation of Mera.
/// @dev This contract is intentionally NOT a trustless bridge. The owner/mint
/// authority must be replaced by an audited bridge or multi-signature/DAO
/// process before any public deployment.
contract MeraScientific is ERC20, ERC20Capped, ERC20Burnable, ERC20Permit, Ownable {
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10 ** 8;

    constructor(address initialOwner)
        ERC20("Mera Scientific Work", "MERA")
        ERC20Capped(MAX_SUPPLY)
        ERC20Permit("Mera Scientific Work")
        Ownable(initialOwner)
    {}

    function decimals() public pure override returns (uint8) {
        return 8;
    }

    /// @notice Controlled mint endpoint for a future audited bridge/issuer.
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Capped)
    {
        super._update(from, to, value);
    }
}
