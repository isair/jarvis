# 🧪 Jarvis Evaluation Report

**Generated:** 2026-04-20 19:44:46

## 📊 TL;DR

**Overall:** 🟢 **248/263 passed (94.3%)** across all categories

| Category | Model | Passed | Failed | Skipped | Pass Rate |
|----------|-------|-------:|-------:|--------:|----------:|
| 🤖 Agent behaviour | `gemma4:e2b` | 91 | 8 | 4 | 🟢 91.9% |
| 🤖 Agent behaviour | `gpt-oss:20b` | 102 | 3 | 0 | 🟢 97.1% |
| 🎤 Intent judge | `gemma4:e2b` (fixed) | 41 | 3 | 0 | 🟢 93.2% |
| 🔍 Tool selection (embedding) | `nomic-embed-text` (fixed) | 5 | 0 | 0 | 🟢 100.0% |
| 🔍 Tool selection (LLM) | `gemma4:e2b` | 5 | 0 | 0 | 🟢 100.0% |
| 🔍 Tool selection (LLM) | `gpt-oss:20b` | 4 | 1 | 0 | 🟢 80.0% |

### 💡 Model Selection Guide

| Model | Best For | Trade-offs |
|-------|----------|------------|
| `gemma4:e2b` | Quick responses, lower RAM usage | May struggle with complex reasoning |
| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |

---

## 🤖 Agent behaviour

> Runs the full agent pipeline against each judge model. Tests are compared side-by-side.

| Test Case | gemma4:e2b | gpt-oss:20b |
|-----------|----------:|----------:|
| 3-turn conversation with topic changes | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent calls webSearch for info queries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent chains search → fetch for details | ❌ 0/1 (0%) | ❌ 0/1 (0%) |
| Agent uses memory + nutrition data | ❌ 0/1 (0%) | ❌ 0/1 (0%) |
| Bad: deflection without attempting answer | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: empty acknowledgment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: generic greeting ignores query | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Diet changed from bulking to cutting | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Diet changed from bulking to cutting-gemma4:e2b | ➖ | 🔸 1/1 XFAIL |
| Diet changed from bulking to cutting-gpt-oss:20b | ➖ | ✅ 1/1 (100%) |
| Enrichment results appear in system message | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Extraction with explicit quantities | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Follow-up references previous turn context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: brief but informative | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: complete weekly forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Handles ambiguous portion descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| LLM uses enrichment-surfaced interests for personalised search | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live greeting: hello | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live greeting: ni hao (Chinese) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live instruction: be more brief | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live instruction: use Celsius | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Live unknown entity: Piranesi (book) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live unknown entity: Possessor (film) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live unknown entity: have-you-heard-of (Piranesi) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live unknown entity: permission-framed (Possessor) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live weather query with real LLM | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Live: LLM checks memory before asking about interests | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live: assistant does not deny having long-term memory | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live: graph-enriched facts surface in reply, no denial | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live: weather query triggers tools | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Location context flows to search queries | ❌ 0/1 (0%) | ❌ 0/1 (0%) |
| LogMealTool stores meals with macros | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Memory enrichment: personalized news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: time-based recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: topic recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: tech news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: time query | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| No deflection: tomorrow weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: weekly rain forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: local business details and user location | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: non-English summary (Turkish) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: relocation plans and employment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: user diet plan and preferred recipe | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: cheeseburger with fries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: chicken with broccoli | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: oatmeal with banana | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Office days changed from Mon/Wed to Mon/Thu | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Office days changed from Mon/Wed to Mon/Thu-gemma4:e2b | ➖ | ✅ 1/1 (100%) |
| Office days changed from Mon/Wed to Mon/Thu-gpt-oss:20b | ➖ | ✅ 1/1 (100%) |
| Reframing: life events framed as facts with temporal context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reframing: requests become knowledge, not interaction descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: assistant self-references (recommendations are not knowledge) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: stale temporal snapshots (weather, time of day) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns NONE for non-food inputs | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns valid JSON with all required fields | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Simple meal baseline (2 boiled eggs) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: explicit tool mention | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: vague go ahead | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: vague just try | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Topic switch: search → weather uses getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Topic switch: weather → store hours uses webSearch | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_active_hot_window_follow_up_accepted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_casual_statement_without_wake_word_rejected | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_date_query_with_date_in_context_returns_none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_digested_tool_result_produces_grounded_reply | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_enrichment_skips_questions_answered_by_context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_first_turn_calls_web_search_not_clarification | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_follow_up_after_correction_calls_web_search | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_followup_naming_place_routes_to_getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_honest_block_when_all_providers_fail | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_identity_query_does_not_trigger_recommendation_engagement_rule | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_identity_query_surfaces_multiple_user_facts_when_present | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_identity_query_surfaces_user_stated_fact_over_past_qa | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_identity_query_with_only_past_qa_returns_none_or_no_false_facts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_echo_claim_overridden_in_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_empty_conversation_returns_empty | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_mixed_summary_filters_noise | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_links_only_payload_produces_honest_cant_read_reply | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_location_query_with_location_in_context_returns_none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_location_query_with_partial_hint_still_routes_sensibly | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| test_no_hint_at_all_still_routes_sensibly | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_no_wake_word_rejected_despite_judge | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_omits_deflection_narration_for_unknown_entity | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_omits_deflection_when_topic_never_resolved | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| test_open_ended_prompt_grounds_in_graph_context_live | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| test_preserves_legitimate_user_preferences | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_raw_text_preserved_in_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_realistic_web_search_payload_is_not_deflected_to_links | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_recommendation_query_still_surfaces_engagement_when_user_facts_present | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_restaurant_recommendation_surfaces_past_cuisine_interest | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_speech_long_after_tts_requires_wake_word | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_stop_during_tts_interrupts_immediately | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_time_query_with_time_in_context_returns_none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_tts_echo_segments_skipped_user_query_extracted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_turn1_possessor_then_turn2_weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_unknown_entity_with_poisoned_diary_still_triggers_web_search_live | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_unrelated_domain_still_returns_none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_unrelated_topics_are_not_welded_into_one_clause | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_user_query_not_confused_with_echo_after_tts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_utterance_started_during_tts_treated_as_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_wake_word_query_after_echo_segments | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_wake_word_query_uses_judge_extraction | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_watch_recommendation_surfaces_recently_discussed_films | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_weather_query_calls_tool_without_asking_for_location | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_weather_query_still_picks_getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_wikipedia_rescues_when_ddg_blocks | ✅ 1/1 (100%) | ✅ 1/1 (100%) |

---

## 🎤 Intent judge

> Pinned to `gemma4:e2b` (the voice intent classifier). Not affected by the judge model.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| alias_after_narrative_context | 1/1 (100%) | ✅ |
| alias_treated_as_wake_word | 1/1 (100%) | ✅ |
| buffer_echo_then_followup_hot_window | 1/1 (100%) | ✅ |
| buried_target_amid_unrelated_chatter | 1/1 (100%) | ✅ |
| buried_target_plural_vague_ref_they | 1/1 (100%) | ✅ |
| buried_target_topicless_question | 1/1 (100%) | ✅ |
| context_synthesis_weather_opinion | 1/1 (100%) | ✅ |
| context_synthesis_with_prior_ambient | 1/1 (100%) | ✅ |
| cross_segment_answer_that_weather | 1/1 (100%) | ✅ |
| cross_segment_answer_that_with_noise | 1/1 (100%) | ✅ |
| cross_segment_answered_that_whisper_variant | 1/1 (100%) | ✅ |
| cross_segment_dinosaur_opinion | 1/1 (100%) | ✅ |
| cross_segment_go_ahead_and_answer | 0/1 (0%) | ❌ |
| cross_segment_hot_window_followup | 1/1 (100%) | ✅ |
| cross_segment_imperative_superseded_by_new_question | 1/1 (100%) | ✅ |
| echo_plus_followup_extracted | 1/1 (100%) | ✅ |
| echo_plus_rejected_similar_plus_wake_retry | 1/1 (100%) | ✅ |
| hot_window_override_topicless_followup | 1/1 (100%) | ✅ |
| hot_window_simple_followup | 1/1 (100%) | ✅ |
| mentioned_in_narrative_past_tense | 1/1 (100%) | ✅ |
| multi_person_vague_reference | 1/1 (100%) | ✅ |
| multi_person_weather_discussion | 0/1 (0%) | ❌ |
| multiple_echoes_then_interrupt | 1/1 (100%) | ✅ |
| no_wake_word_casual_speech | 1/1 (100%) | ✅ |
| no_wake_word_in_buffer | 1/1 (100%) | ✅ |
| stop_command_during_tts | 1/1 (100%) | ✅ |
| test_hot_window_mode_indicated_in_prompt | 1/1 (100%) | ✅ |
| test_old_query_not_re_extracted | 1/1 (100%) | ✅ |
| test_processed_segment_not_reextracted | 1/1 (100%) | ✅ |
| test_returns_none_when_ollama_unavailable | 1/1 (100%) | ✅ |
| test_system_prompt_has_echo_guidance | 1/1 (100%) | ✅ |
| test_tts_text_included_for_echo_detection | 1/1 (100%) | ✅ |
| user_followup_statement_after_question_nihilism | 0/1 (0%) | ❌ |
| wake_word_after_narrative_addresses_assistant | 1/1 (100%) | ✅ |
| wake_word_command_timer | 1/1 (100%) | ✅ |
| wake_word_mid_sentence | 1/1 (100%) | ✅ |
| wake_word_open_imperative_give_me_advice | 1/1 (100%) | ✅ |
| wake_word_open_imperative_say_something | 1/1 (100%) | ✅ |
| wake_word_open_imperative_surprise_me | 1/1 (100%) | ✅ |
| wake_word_open_imperative_tell_me_a_joke | 1/1 (100%) | ✅ |
| wake_word_open_imperative_tell_me_anything | 1/1 (100%) | ✅ |
| wake_word_simple_question | 1/1 (100%) | ✅ |
| wake_word_statement_remember | 1/1 (100%) | ✅ |
| wake_word_trailing_after_named_entity | 1/1 (100%) | ✅ |

---

## 🔍 Tool selection

### Embedding strategy

> Pinned to `nomic-embed-text` (embedding-based filter). Not affected by the judge model.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| location weather query selects getWeather and few others | 1/1 (100%) | ✅ |
| meal logging selects logMeal and few others | 1/1 (100%) | ✅ |
| meal recall selects fetchMeals and few others | 1/1 (100%) | ✅ |
| weather query selects getWeather and few others | 1/1 (100%) | ✅ |
| web search query selects webSearch and few others | 1/1 (100%) | ✅ |

### LLM strategy (default)

> Exercises the default `llm` router strategy against each supported chat model.

| Test Case | gemma4:e2b | gpt-oss:20b |
|-----------|----------:|----------:|
| weather query selects getWeather and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| location weather query selects getWeather and few others | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| meal logging selects logMeal and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| meal recall selects fetchMeals and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| web search query selects webSearch and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |

---

### 📖 Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully passed (100% pass rate) |
| ⚠️ | Partial pass (some runs failed) |
| ❌ | Fully failed (0% pass rate) |
| ⏭️ | Skipped (missing dependencies) |
| 🔸 | Expected failure (known limitation) |
| 🎉 | Unexpectedly passed (bug fixed!) |
| ➖ | Not run for this model |

*Report generated by Jarvis eval suite*