# Security Center — Risk Weight Documentation (Phase 1)

Scores are **risk scores** (0 = better, 100 = worse).

## Category weights (`security/config/security_config.json` → `weights`)

Each category accumulates additive points from signals, then clamps to 0–100.

### remote_access
| Signal | Points |
|--------|--------|
| remote_tool_present | 40 |
| rdp_enabled | 25 |
| winrm_running | 15 |
| ssh_listening | 15 |
| smb_session_active | 10 |
| unknown_listener_all_interfaces | 20 |
| unknown_outbound | 15 |

### persistence
| Signal | Points |
|--------|--------|
| new_run_key | 20 |
| new_task | 20 |
| new_service | 20 |
| hosts_modified | 15 |
| wmi_nondefault | 30 |
| ifeo_debugger | 35 |
| appinit_dll | 35 |
| winlogon_shell_changed | 40 |

### account
| Signal | Points |
|--------|--------|
| blank_password_admin | 35 |
| new_admin | 40 |
| new_user | 20 |
| guest_enabled | 25 |

### network
| Signal | Points |
|--------|--------|
| new_listening_port | 15 |
| unknown_established | 20 |
| firewall_disabled | 30 |

### malware_indicators
| Signal | Points |
|--------|--------|
| unsigned_temp_exe | 25 |
| trusted_name_bad_path | 30 |
| remote_tool_keyword | 35 |
| lolbin_suspicious | 20 |
| hash_changed | 25 |

### hygiene
| Signal | Points |
|--------|--------|
| secure_boot_off | 20 |
| defender_and_av_off | 30 |
| firewall_off | 25 |
| bitlocker_unknown_or_off | 10 |
| min_password_zero | 20 |

### forensic_visibility
| Signal | Points |
|--------|--------|
| security_log_unavailable | 25 |
| sysmon_absent | 15 |
| task_scheduler_op_disabled | 10 |
| audit_logon_unknown | 15 |
| prefetch_unavailable | 5 |

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
