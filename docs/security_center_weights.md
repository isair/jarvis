# Security Center — Risk Weight Documentation (Phase 1)

Scores are **risk scores** (0 = better, 100 = worse).

## Category weights (`security/config/security_config.json` → `weights`)

Each category accumulates additive points from signals, then clamps to 0–100.

Keys marked **reserved (Phase 1)** exist in config/defaults for forward compatibility but are **not yet applied** by `risk.py` scoring. Do not treat them as active until wired.

### remote_access
| Signal | Points | Status |
|--------|--------|--------|
| remote_tool_present | 40 | active |
| rdp_enabled | 25 | active |
| winrm_running | 15 | active |
| ssh_listening | 15 | active |
| smb_session_active | 10 | active |
| unknown_listener_all_interfaces | 20 | active |
| unknown_outbound | 15 | reserved (Phase 1) |

### persistence
| Signal | Points | Status |
|--------|--------|--------|
| new_run_key | 20 | active |
| new_task | 20 | active |
| new_service | 20 | active |
| hosts_modified | 15 | active |
| wmi_nondefault | 30 | active |
| ifeo_debugger | 35 | active |
| appinit_dll | 35 | active |
| winlogon_shell_changed | 40 | active |

### account
| Signal | Points | Status |
|--------|--------|--------|
| blank_password_admin | 35 | active |
| new_admin | 40 | active |
| new_user | 20 | active |
| guest_enabled | 25 | active |

### network
| Signal | Points | Status |
|--------|--------|--------|
| new_listening_port | 15 | active |
| unknown_established | 20 | reserved (Phase 1) |
| firewall_disabled | 30 | active |

### malware_indicators
| Signal | Points | Status |
|--------|--------|--------|
| unsigned_temp_exe | 25 | active |
| trusted_name_bad_path | 30 | active |
| remote_tool_keyword | 35 | active |
| lolbin_suspicious | 20 | reserved (Phase 1) |
| hash_changed | 25 | reserved (Phase 1) |

### hygiene
| Signal | Points | Status |
|--------|--------|--------|
| secure_boot_off | 20 | active |
| defender_and_av_off | 30 | active |
| firewall_off | 25 | active |
| bitlocker_unknown_or_off | 10 | active |
| min_password_zero | 20 | active |

### forensic_visibility
| Signal | Points | Status |
|--------|--------|--------|
| security_log_unavailable | 25 | active |
| sysmon_absent | 15 | active |
| task_scheduler_op_disabled | 10 | active |
| audit_logon_unknown | 15 | active |
| prefetch_unavailable | 5 | reserved (Phase 1) |

## Overall blend (`overall_blend`)

| Category | Weight |
|----------|--------|
| remote_access | 0.22 |
| persistence | 0.18 |
| account | 0.15 |
| network | 0.12 |
| malware_indicators | 0.15 |
| hygiene | 0.10 |
| forensic_visibility | 0.08 |

**Important:** A low overall score is **not** a cleanliness certificate when forensic visibility is poor.
