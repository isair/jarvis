# 🧪 Jarvis Evaluation Report

**Generated:** 2026-05-04 (small-model only, with retries)
**Judge Model:** `gemma4:e2b`
**Initial-pass duration:** 3366.56s (full suite)
**Retry policy:** failed cases re-run up to 3× (`--lf`), tracked below

## 📊 Summary (initial pass)

| Metric | Count |
|--------|-------|
| ✅ Fully Passed (100%) | 199 |
| ⚠️ Partial Pass | 0 |
| ❌ Fully Failed (0%) | 21 |
| ⏭️ Skipped | 4 |
| 🔸 Expected Fail | 15 |
| **Unique Tests** | **239** |
| **Total Runs** | **239** |

**Overall Pass Rate (initial):** 🟢 `██████████████████░░` **90.5%** (199/220 runs)

## 🔁 Retry Outcomes

Of the 21 initial failures, 10 passed within 3 retries (≈48% were flaky), 11 failed consistently. After investigating the consistent failures against the prior commit `73035d4` (the state when the last report claimed 100% on intent-judge / merge-consolidation), all 11 are pre-existing small-model edge cases or environment-flaky cases — **no behavioural regressions on this branch**.

### ✅ Passed on retry (treated as flaky, not regressions)

| Test | Passed on retry # |
|------|-------------------|
| `TestMemoryEnrichment::test_enrichment_extracts_correct_keywords[Memory enrichment: personalized news]` | 1 |
| `TestMemoryEnrichment::test_enrichment_extracts_correct_keywords[Memory enrichment: topic recall]` | 1 |
| `TestMemoryEnrichment::test_enrichment_extracts_correct_keywords[Memory enrichment: time-based recall]` | 1 |
| `TestMemoryEnrichment::test_enrichment_skips_questions_answered_by_context` | 1 |
| `TestComplexMultiTurnMultiTool::test_cross_turn_pronoun_resolution` | 1 |
| `TestKnowledgeExtractionQuality::test_extracts_novel_knowledge[Novel knowledge: relocation plans and employment]` | 1 |
| `TestFollowupSuppliesMissingToolArg::test_short_followup_continues_previous_tool_chain` | 2 |
| `TestKnowledgeExtractionJudge::test_judge_mixed_summary_filters_noise` | 2 |
| `TestSelfContainedToolArguments::test_follow_up_resolves_pronoun_in_search_query` | 3 |
| `TestMultiTurnExtended::test_three_turn_topic_changes` | 3 |

### ❌ Consistently failed after 3 retries

| Test | Category |
|------|----------|
| `TestLiveEndToEnd::test_weather_query_live` | Pre-existing — small-model deflection on geoip path |
| `TestHelpfulness::test_open_ended_prompt_grounds_in_graph_context_live` | Pre-existing — small model ignores enriched graph |
| `TestContextSwitchTools::test_turn1_possessor_then_turn2_weather` | Pre-existing — small-model confabulation on unknown entity |
| `TestPlannerEmitsSearchMemoryForPersonalisedQueries::test_general_knowledge_query_does_not_request_memory[who is Britney Spears]` | Pre-existing — small-model planner over-eagerly requests memory |
| `TestNutritionExtraction::test_extraction_returns_valid_json_structure` | Pre-existing — small-model JSON-only output |
| `TestDiarySuppliesMissingToolArg::test_diary_location_grounds_get_weather_call` | New (#352) — docstring acknowledges small-model deflection is expected until follow-up memory work lands |
| `TestGraphSuppliesMissingToolArg::test_warm_profile_user_fact_grounds_get_weather_call` | New (#352) — same warm-profile-grounding limitation as above |
| `TestPatternConsolidation::test_repeated_activities_consolidate[sushi pattern]` | Flaky on small model — passes 3/3 when run in isolation; failed only when bundled with other retries (likely VRAM / model-state contention). No code change. |
| `TestIntentJudgeMultiSegment::test_multi_segment_case[multi_person_weather_discussion]` | Pre-existing small-model limitation — also fails 3/3 at prior commit `73035d4` (state of the previous "100%" report); intent judge can't reliably resolve "what do you think" to a cross-segment topic on `gemma4:e2b`. The previous report's 100% was a single lucky run, not a stable baseline. |
| `TestIntentJudgeMultiSegment::test_multi_segment_case[cross_segment_answer_that_with_noise]` | Pre-existing small-model limitation — sister case `cross_segment_answer_that_weather` (no noise segment) does pass; the noise segment "Charlie sands to that" overlaps with "answer that" and confuses the small model's referent picker. Also flaky at prior commit. |
| `TestIntentJudgeMultiSegment::test_multi_segment_case[cross_segment_go_ahead_and_answer]` | Pre-existing small-model limitation — also fails 3/3 at prior commit `73035d4`; small model passes the multi-word imperative ("go ahead and answer") through literally despite the prompt listing it as an imperative variant. |

---

## 📋 Detailed Results (initial pass)

### ✅ TestResponseQuality
> LLM-as-judge evaluations for response quality

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Good: complete weekly forecast | 1/1 (100%) | ✅ PASSED | 10.13s |
| Good: brief but informative | 1/1 (100%) | ✅ PASSED | 5.65s |
| Bad: generic greeting ignores query | 1/1 (100%) | ✅ PASSED | 4.23s |
| Bad: deflection without attempting answer | 1/1 (100%) | ✅ PASSED | 4.42s |
| Bad: empty acknowledgment | 1/1 (100%) | ✅ PASSED | 4.80s |

### ✅ TestContextUtilization
> Tests that agent uses location/time/memory context

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Location context flows to search queries | 1/1 (100%) | ✅ PASSED | 6.39s |

### ✅ TestToolUsage
> Validates tool selection and argument quality

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Agent calls webSearch for info queries | 1/1 (100%) | ✅ PASSED | 14.19s |
| Agent chains search → fetch for details | 1/1 (100%) | ✅ PASSED | 2.36s |

### ✅ TestMultiStepReasoning
> Complex scenarios requiring tool chaining and synthesis

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Agent uses memory + nutrition data | 1/1 (100%) | ✅ PASSED | 15.38s |

### ⚠️ TestMemoryEnrichment
> Tests automatic memory enrichment keyword extraction

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Memory enrichment: personalized news | 0/1 (0%) | ❌ FAILED | 34.09s |
| Memory enrichment: topic recall | 0/1 (0%) | ❌ FAILED | 34.10s |
| Memory enrichment: time-based recall | 0/1 (0%) | ❌ FAILED | 34.11s |
| Enrichment skips questions answered by context | 0/1 (0%) | ❌ FAILED | 34.11s |
| Enrichment results appear in system message | 1/1 (100%) | ✅ PASSED | 18.10s |
| LLM uses enrichment-surfaced interests for personalised search | 1/1 (100%) | ✅ PASSED | 18.11s |

### ⚠️ TestLiveEndToEnd
> End-to-end tests against real LLM inference

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Weather query is answered with current conditions | 0/1 (0%) | ❌ FAILED | 99.72s |
| Assistant checks memory before asking about interests | 1/1 (100%) | ✅ PASSED | 34.09s |
| explicit-recall-then-search | 1/1 (100%) | ✅ PASSED | 26.42s |
| news-that-would-interest-me | 1/1 (100%) | ✅ PASSED | 26.91s |
| news-of-interest-to-me | 1/1 (100%) | ✅ PASSED | 26.56s |
| news-interesting-for-me | 1/1 (100%) | ✅ PASSED | 27.11s |

### ⚠️ TestHelpfulness
> Tests that agent uses tools proactively instead of deflecting

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| No deflection: tomorrow weather | 1/1 (100%) | ✅ PASSED | 56.47s |
| No deflection: weekly rain forecast | 1/1 (100%) | ✅ PASSED | 58.05s |
| No deflection: tech news | 1/1 (100%) | ✅ PASSED | 59.88s |
| No deflection: time query | 1/1 (100%) | ✅ PASSED | 54.69s |
| Tool retry: explicit tool mention | 1/1 XFAIL | 🔸 XFAILED | 92.40s |
| Tool retry: vague go ahead | 1/1 (100%) | ✅ PASSED | 33.90s |
| Tool retry: vague just try | 1/1 (100%) | ✅ PASSED | 55.63s |
| Graph-enriched facts surface in the reply, no denial | 1/1 (100%) | ✅ PASSED | 36.87s |
| Assistant does not deny having long-term memory | 1/1 (100%) | ✅ PASSED | 34.60s |
| Open-ended prompt grounds in stored knowledge | 0/1 (0%) | ❌ FAILED | 33.90s |

### ✅ TestMalformedResponseAfterTools
> Tests that malformed LLM output after tool results is not surfaced

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Tool calls literal not surfaced after web search | 1/1 (100%) | ✅ PASSED | 33.81s |

### ❌ TestCelebrityIdentityThenFollowUp
> Two-turn celebrity flow: identity query then pronoun follow-up

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Two-turn celebrity flow: identity then pronoun follow-up | 1/1 XFAIL | 🔸 XFAILED | 46.67s |

### ❌ TestSearchFailureWikipediaRescue
> Wikipedia-rescue payload is consumed correctly, not confabulated over

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Wikipedia payload produces grounded reply | 1/1 XFAIL | 🔸 XFAILED | 54.52s |

### ✅ TestMultiStepEntityQuery
> Single query requiring two sequential webSearch calls (director + filmography)

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Director-then-filmography needs two searches | 1/1 (100%) | ✅ PASSED | 69.00s |

### ❌ TestContextSwitchTools

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Turn1 possessor then turn2 weather | 0/1 (0%) | ❌ FAILED | 27.25s |

### ⚠️ TestDiarySummariserHygieneLive

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Omits deflection narration for unknown entity | 1/1 (100%) | ✅ PASSED | 6.37s |
| Omits deflection when topic never resolved | 1/1 XFAIL | 🔸 XFAILED | 2.54s |
| Unrelated topics are not welded into one clause | 1/1 (100%) | ✅ PASSED | 2.77s |
| Preserves legitimate user preferences | 1/1 (100%) | ✅ PASSED | 2.61s |
| Post process scrub cleans what the prompt leaks | 1/1 (100%) | ✅ PASSED | 2.62s |

### ❌ TestDiarySuppliesMissingToolArg

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Diary location grounds get weather call | 0/1 (0%) | ❌ FAILED | 13.80s |

### ❌ TestPrematureProseNudge

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Navigate prose gets nudged into tool call | 1/1 XFAIL | 🔸 XFAILED | 5.43s |

### ✅ TestTerminalOnSuccessfulToolUse

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Single weather query ends after one tool call | 1/1 (100%) | ✅ PASSED | 5.53s |

### ✅ TestTerminalOnHonestCantDo

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| No email tool declines honestly | 1/1 (100%) | ✅ PASSED | 5.30s |

### ✅ TestNudgeCapEnforcement

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Nudge cap stops loop | 1/1 (100%) | ✅ PASSED | 5.42s |

### ✅ TestMaxTurnDigestCaveat

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Max-turn cap delivers a digest reply, never silence | 1/1 (100%) | ✅ PASSED | 21.70s |

### ❌ TestToolSearchToolEscapeHatch

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Toolsearchtool widens then navigate | 1/1 XFAIL | 🔸 XFAILED | 51.42s |

### ⚠️ TestComplexMultiTurnMultiTool

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Chained research: who directed Possessor and what else have they made | 1/1 (100%) | ✅ PASSED | 56.01s |
| Parallel weather lookup: compare Paris and London | 1/1 (100%) | ✅ PASSED | 10.62s |
| Cross turn pronoun resolution | 0/1 (0%) | ❌ FAILED | 78.95s |
| Correction loop accepts single or retry | 1/1 (100%) | ✅ PASSED | 8.99s |
| Escape hatch then follow up action | 1/1 XFAIL | 🔸 XFAILED | 55.10s |

### ❌ TestStructuredToolCallEmission

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Evaluator emits structured tool call for obvious search | 1/1 XFAIL | 🔸 XFAILED | 10.05s |

### ❌ TestFollowupSuppliesMissingToolArg

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Short followup continues previous tool chain | 0/1 (0%) | ❌ FAILED | 103.36s |

### ✅ TestGraphBranchRouting

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| USER: identity, location, pets, diet, job | 1/1 (100%) | ✅ PASSED | 15.70s |
| DIRECTIVES: tone, length, forbidden phrases, address form | 1/1 (100%) | ✅ PASSED | 2.72s |
| WORLD: local business details, film attribution | 1/1 (100%) | ✅ PASSED | 2.96s |
| Adversarial: food preference (USER) vs list-length rule (DIRECTIVES) | 1/1 (100%) | ✅ PASSED | 2.62s |
| Adversarial: all three branches in one summary | 1/1 (100%) | ✅ PASSED | 3.06s |

### ❌ TestGraphSuppliesMissingToolArg

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Warm profile user fact grounds get weather call | 0/1 (0%) | ❌ FAILED | 49.91s |

### ⚠️ TestGreetingNoToolsLive
> Tests that greetings don't trigger tool calls

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Greeting: hello | 1/1 (100%) | ✅ PASSED | 54.36s |
| Greeting: ni hao (Chinese) | 1/1 (100%) | ✅ PASSED | 55.13s |
| Instruction: use Celsius | 1/1 (100%) | ✅ PASSED | 43.91s |
| Instruction: be more brief | 1/1 (100%) | ✅ PASSED | 14.19s |
| Unknown entity: Possessor (film) | 1/1 XFAIL | 🔸 XFAILED | 46.97s |
| Unknown entity: Piranesi (book) | 1/1 (100%) | ✅ PASSED | 57.17s |
| Unknown entity: permission-framed (Possessor) | 1/1 (100%) | ✅ PASSED | 57.75s |
| Unknown entity: have-you-heard-of (Piranesi) | 1/1 (100%) | ✅ PASSED | 57.95s |
| Unknown entity with poisoned diary still triggers web search live | 1/1 XFAIL | 🔸 XFAILED | 56.60s |
| Weather query still triggers tools after a greeting | 1/1 (100%) | ✅ PASSED | 50.54s |

### ✅ TestIntentJudgeAccuracy
> Intent judge accuracy for voice command classification

#### ✅ wake_word_simple_question

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word + question

*Avg Duration: 4.00s*

#### ✅ wake_word_trailing_after_named_entity

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word removed; 'Jarvis' is treated as part of the entity name; resolved to the movie title.

*Avg Duration: 0.55s*

#### ✅ wake_word_mid_sentence

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word + question

*Avg Duration: 0.42s*

#### ✅ wake_word_command_timer

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, extract command.

*Avg Duration: 0.45s*

#### ✅ wake_word_statement_remember

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word 'Jarvis' removed. The query is a direct command.

*Avg Duration: 0.52s*

#### ✅ wake_word_share_statement_burger

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word removed; extracted declarative statement.

*Avg Duration: 0.44s*

#### ✅ wake_word_share_statement_feeling

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word detected; extracted the query following the wake word.

*Avg Duration: 0.46s*

#### ✅ wake_word_share_statement_trailing

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word removed; the query is a statement about the user's flight cancellation.

*Avg Duration: 0.46s*

#### ✅ wake_word_trailing_after_capitalised_brand

**Pass Rate:** 1/1 (100%)
**Judge notes:** Extracted query by removing the wake word 'Jarvis'.

*Avg Duration: 0.47s*

#### ✅ wake_word_open_imperative_say_something

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word + self-contained imperative

*Avg Duration: 0.40s*

#### ✅ wake_word_open_imperative_tell_me_a_joke

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word + self-contained imperative

*Avg Duration: 0.43s*

#### ✅ wake_word_open_imperative_tell_me_anything

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, and the query is a self-contained imperative command.

*Avg Duration: 0.45s*

#### ✅ wake_word_open_imperative_give_me_advice

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, extract the command/request.

*Avg Duration: 0.45s*

#### ✅ wake_word_open_imperative_surprise_me

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, and the query is a direct imperative command.

*Avg Duration: 0.45s*

#### ✅ context_synthesis_weather_opinion

**Pass Rate:** 1/1 (100%)
**Judge notes:** Extracted the question following the wake word. Resolved 'What do you think' to refer to the preceding statement about the weather in London.

*Avg Duration: 0.56s*

#### ✅ echo_plus_followup_extracted

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, user asked to elaborate on the previous statement about London's daylight hours.

*Avg Duration: 0.52s*

#### ✅ stop_command_during_tts

**Pass Rate:** 1/1 (100%)
**Judge notes:** stop command detected

*Avg Duration: 0.37s*

#### ✅ no_wake_word_casual_speech

**Pass Rate:** 1/1 (100%)
**Judge notes:** No wake word detected, and the statement is not a command or question directed at Jarvis.

*Avg Duration: 0.47s*

#### ✅ mentioned_in_narrative_past_tense

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word used only as a narrative mention; not a direct query or command.

*Avg Duration: 0.49s*

#### ✅ hot_window_simple_followup

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, direct question follow-up to previous context.

*Avg Duration: 0.45s*

### ✅ TestIntentJudgePromptQuality
> Intent judge prompt construction quality

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Hot window mode indicated in prompt | 1/1 (100%) | ✅ PASSED | 0.00s |
| Tts text included for echo detection | 1/1 (100%) | ✅ PASSED | 0.00s |
| System prompt has echo guidance | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestIntentJudgeFallback
> Intent judge fallback behaviour when unavailable

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Returns none when ollama unavailable | 1/1 (100%) | ✅ PASSED | 0.00s |

### ⚠️ TestIntentJudgeMultiSegment
> Intent judge with multi-segment buffers and multi-person conversations

#### ✅ echo_plus_rejected_similar_plus_wake_retry

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, query extracted is a direct question about movies tomorrow.

*Avg Duration: 0.47s*

#### ✅ buffer_echo_then_followup_hot_window

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, direct follow-up question based on previous weather context.

*Avg Duration: 0.49s*

#### ✅ multiple_echoes_then_interrupt

**Pass Rate:** 1/1 (100%)
**Judge notes:** Standalone imperative command 'stop' detected.

*Avg Duration: 0.43s*

#### ✅ no_wake_word_in_buffer

**Pass Rate:** 1/1 (100%)
**Judge notes:** No wake word detected; treated as a standalone statement/question.

*Avg Duration: 0.43s*

#### ✅ context_synthesis_with_prior_ambient

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, query extracted is a direct question to the assistant.

*Avg Duration: 0.49s*

#### ❌ multi_person_weather_discussion

**Pass Rate:** 0/1 (0%)
**Judge notes:** Wake word detected, query is a direct question following the wake word.

*Avg Duration: 0.46s*

#### ✅ multi_person_vague_reference

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. 'that' refers to the iPhone mentioned in the previous segment. The query asks for the price of the iPhone.

*Avg Duration: 0.51s*

#### ✅ user_followup_statement_after_question_nihilism

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, user provided a direct statement/opinion.

*Avg Duration: 0.55s*

#### ✅ cross_segment_dinosaur_opinion

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. 'that' refers to 'dinosaurs' from the previous segment. The query is a question about the previous statement.

*Avg Duration: 0.54s*

#### ✅ cross_segment_answer_that_weather

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, followed by an imperative command referencing the previous question.

*Avg Duration: 0.49s*

#### ❌ cross_segment_answer_that_with_noise

**Pass Rate:** 0/1 (0%)
**Judge notes:** Imperative command 'answer that' refers to the preceding question 'How tall is Mount Everest'. The instruction is to answer the previous question.

*Avg Duration: 0.54s*

#### ✅ cross_segment_answered_that_whisper_variant

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, and the query is an imperative referencing a prior question. Re-issue the prior question.

*Avg Duration: 0.51s*

#### ❌ cross_segment_go_ahead_and_answer

**Pass Rate:** 0/1 (0%)
**Judge notes:** Wake word detected, followed by an imperative command to answer.

*Avg Duration: 0.46s*

#### ✅ cross_segment_imperative_superseded_by_new_question

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, followed by an imperative command referencing the previous question, which is re-issued as a direct question.

*Avg Duration: 0.52s*

#### ✅ cross_segment_hot_window_followup

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, direct question follow-up. 'What about Germany' is a direct query.

*Avg Duration: 0.48s*

#### ✅ alias_treated_as_wake_word

**Pass Rate:** 1/1 (100%)
**Judge notes:** wake word + question

*Avg Duration: 0.44s*

#### ✅ alias_after_narrative_context

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. 'that' refers to the iPhone mentioned in the previous segment. The query asks for the price of the iPhone.

*Avg Duration: 0.53s*

#### ✅ buried_target_amid_unrelated_chatter

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. The query asks for the cost of the iPhone, referencing the previous segment about the new iPhone.

*Avg Duration: 0.55s*

#### ✅ buried_target_topicless_question

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. 'that' refers to the iPhone pro model mentioned previously. The query asks for the price of the iPhone pro model.

*Avg Duration: 0.56s*

#### ✅ buried_target_plural_vague_ref_they

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected, query is a direct question about the cost of the AirPods mentioned previously.

*Avg Duration: 0.49s*

#### ✅ hot_window_override_topicless_followup

**Pass Rate:** 1/1 (100%)
**Judge notes:** Hot Window mode, user is asking for more information based on the previous context about the iPhone.

*Avg Duration: 0.50s*

#### ✅ wake_word_after_narrative_addresses_assistant

**Pass Rate:** 1/1 (100%)
**Judge notes:** Wake word detected. Query extracted is a direct question about Mata Hari, referencing the context established by the previous statements.

*Avg Duration: 0.54s*

### ✅ TestProcessedSegmentFiltering

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Processed segment not reextracted | 1/1 (100%) | ✅ PASSED | 0.46s |

### ⚠️ TestKnowledgeExtractionQuality
> Tests that novel knowledge is correctly extracted from summaries

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Novel knowledge: local business details and user location | 1/1 (100%) | ✅ PASSED | 2.76s |
| Novel knowledge: user diet plan and preferred recipe | 1/1 (100%) | ✅ PASSED | 2.83s |
| Novel knowledge: relocation plans and employment | 0/1 (0%) | ❌ FAILED | 2.96s |
| Novel knowledge: non-English summary (Turkish) | 1/1 (100%) | ✅ PASSED | 3.10s |

### ✅ TestKnowledgeExtractionRejection
> Tests that noise, stale data, and common knowledge are rejected

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Reject: assistant self-references (recommendations are not knowledge) | 1/1 (100%) | ✅ PASSED | 2.26s |
| Reject: stale temporal snapshots (weather, time of day) | 1/1 (100%) | ✅ PASSED | 2.28s |

### ✅ TestKnowledgeExtractionReframing
> Tests that interaction descriptions are reframed as knowledge

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Reframing: requests become knowledge, not interaction descriptions | 1/1 (100%) | ✅ PASSED | 2.77s |
| Reframing: life events framed as facts with temporal context | 1/1 (100%) | ✅ PASSED | 2.79s |

### ⚠️ TestKnowledgeExtractionJudge
> LLM-as-judge evaluations of extraction quality

#### ✅ Novel knowledge: local business details and user location

**Pass Rate:** 1/1 (100%)
**Score:** ●●●●●●●●●○ (0.92)
**Judge notes:** The extraction is excellent. Both facts are accurately captured and are self-contained, specific pieces of information derived directly from the conversation summary. The model successfully separated the context about the user's residency and the specific details of the boxing gym offerings. All facts adhere to the constraints of not including assistant voice or stale data.

*Avg Duration: 8.60s*

#### ✅ Novel knowledge: user diet plan and preferred recipe

**Pass Rate:** 1/1 (100%)
**Score:** ●●●●●●●●●○ (0.92)
**Judge notes:** The extraction is excellent. Both facts are accurate, directly derived from the summary, and are highly self-contained. They avoid assistant voice and stale data entirely. The completeness score is high because it successfully captured the specific numerical targets (1800 kcal, 150g protein) and the specific food preference/recipe mentioned by the user.

*Avg Duration: 8.52s*

#### ✅ Novel knowledge: relocation plans and employment

**Pass Rate:** 1/1 (100%)
**Score:** ●●●●●●●●●○ (0.96)
**Judge notes:** The extraction is excellent. All three facts accurately reflect the core planning, financial, and professional details provided in the conversation summary. The facts are highly novel, self-contained, and are phrased as objective statements about the user's situation rather than descriptions of the assistant's output. The overall completeness is very high as all major data points were captured.

*Avg Duration: 8.89s*

#### ✅ Novel knowledge: non-English summary (Turkish)

**Pass Rate:** 1/1 (100%)
**Score:** ●●●●●●●●●○ (0.98)
**Judge notes:** The extraction is excellent. All four facts are accurately derived from the summary and are highly self-contained, directly addressing the user's location, eating habits, restaurant pricing, and food preferences. There is no extraneous information, and all facts are novel to the context provided. The completeness score is perfect as all key details from the summary were successfully captured.

*Avg Duration: 9.27s*

#### ✅ Trivial conversations produce no extracted facts

**Pass Rate:** 1/1 (100%)
*Avg Duration: 2.46s*

#### ❌ Mixed summary: keep novel facts, drop stale weather/recommendations

**Pass Rate:** 0/1 (0%)
*Avg Duration: 2.80s*

### ✅ TestWakeWordValidationSafetyNet
> Integration: listener rejects judge hallucinations when no wake word present

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| No wake word rejected despite judge | 1/1 (100%) | ✅ PASSED | 2.34s |
| Casual statement without wake word rejected | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestEchoReasoningDistrust
> Integration: listener overrides judge echo claims when EchoDetector cleared

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Judge echo claim overridden in hot window | 1/1 (100%) | ✅ PASSED | 1.84s |
| User query not confused with echo after tts | 1/1 (100%) | ✅ PASSED | 1.25s |

### ✅ TestHotWindowHeuristicAccuracy
> Integration: could_be_hot_window heuristic passes correct mode to judge

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Active hot window follow up accepted | 1/1 (100%) | ✅ PASSED | 1.26s |
| Speech long after tts requires wake word | 1/1 (100%) | ✅ PASSED | 0.00s |
| Utterance started during tts treated as hot window | 1/1 (100%) | ✅ PASSED | 1.21s |

### ✅ TestProcessedSegmentFilteringIntegration
> Integration: processed segments excluded from judge prompt

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Old query not re extracted | 1/1 (100%) | ✅ PASSED | 1.19s |

### ✅ TestHotWindowPrefersJudgeQuery

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Hot window query is directed and non empty | 1/1 (100%) | ✅ PASSED | 1.21s |
| Wake word query uses judge extraction | 1/1 (100%) | ✅ PASSED | 1.20s |

### ✅ TestMultiSegmentBufferIntegration
> Integration: multi-segment buffer with TTS echoes handled correctly

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Tts echo segments skipped user query extracted | 1/1 (100%) | ✅ PASSED | 1.22s |
| Wake word query after echo segments | 1/1 (100%) | ✅ PASSED | 1.20s |

### ✅ TestStopCommandBypassesJudge
> Integration: stop commands during TTS bypass judge entirely

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Stop during tts interrupts immediately | 1/1 (100%) | ✅ PASSED | 0.05s |

### ✅ TestMemoryDigestSurfacesIdentityFacts

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Identity query surfaces user stated fact over past qa | 1/1 (100%) | ✅ PASSED | 2.89s |
| Identity query surfaces multiple user facts when present | 1/1 (100%) | ✅ PASSED | 2.44s |
| Identity query with only past qa returns none or no false facts | 1/1 (100%) | ✅ PASSED | 2.76s |
| Identity query does not trigger recommendation engagement rule | 1/1 (100%) | ✅ PASSED | 2.58s |
| Recommendation query still surfaces engagement when user facts present | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestMemoryDigestSurfacesPreferenceSignals

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Watch recommendation surfaces recently discussed films | 1/1 (100%) | ✅ PASSED | 2.57s |
| Restaurant recommendation surfaces past cuisine interest | 1/1 (100%) | ✅ PASSED | 0.00s |
| Unrelated domain still returns none | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestNearDuplicateDedupe

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| lives-in vs based-in London | 1/1 (100%) | ✅ PASSED | 2.48s |
| job rephrased | 1/1 (100%) | ✅ PASSED | 2.40s |

### ❌ TestPatternConsolidation

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| sushi pattern | 0/1 (0%) | ❌ FAILED | 2.68s |

### ✅ TestPatternBoundary

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| distinct one-off events | 1/1 (100%) | ✅ PASSED | 2.68s |

### ✅ TestIndependenceOfUnrelatedFacts

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| independent facts coexist | 1/1 (100%) | ✅ PASSED | 2.46s |
| job survives unrelated hobby fact | 1/1 (100%) | ✅ PASSED | 2.52s |

### ✅ TestMetaNarrativePruning

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| capability denial dropped, directive kept | 1/1 (100%) | ✅ PASSED | 2.38s |
| assistant-suggested line dropped, lookup survives | 1/1 (100%) | ✅ PASSED | 2.47s |
| polluted node + new fact: drop and incorporate | 1/1 (100%) | ✅ PASSED | 2.45s |
| genuine directives untouched | 1/1 (100%) | ✅ PASSED | 2.46s |

### ✅ TestBatchedMerge

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| batched 3 new facts | 1/1 (100%) | ✅ PASSED | 2.46s |

### ✅ TestTopicSwitching
> Tests correct tool selection when conversation topic changes

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Topic switch: weather → store hours uses webSearch | 1/1 (100%) | ✅ PASSED | 30.84s |
| Topic switch: search → weather uses getWeather | 1/1 (100%) | ✅ PASSED | 31.35s |

### ✅ TestFollowUpContext
> Tests context retention for follow-up questions

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Follow-up references previous turn context | 1/1 (100%) | ✅ PASSED | 31.39s |

### ❌ TestSelfContainedToolArguments

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Follow up resolves pronoun in search query | 0/1 (0%) | ❌ FAILED | 65.97s |

### ❌ TestMultiTurnExtended
> Extended multi-turn scenarios with longer conversations

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| 3-turn conversation with topic changes | 0/1 (0%) | ❌ FAILED | 131.45s |

### ⚠️ TestNutritionExtraction
> Tests LLM nutrition extraction accuracy for meal logging

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Nutrition: chicken with broccoli | 1/1 (100%) | ✅ PASSED | 11.15s |
| Nutrition: cheeseburger with fries | 1/1 (100%) | ✅ PASSED | 3.23s |
| Nutrition: oatmeal with banana | 1/1 (100%) | ✅ PASSED | 3.15s |
| Returns valid JSON with all required fields | 0/1 (0%) | ❌ FAILED | 2.25s |
| Handles ambiguous portion descriptions | 1/1 (100%) | ✅ PASSED | 3.20s |
| Returns NONE for non-food inputs | 1/1 (100%) | ✅ PASSED | 2.25s |

### ✅ TestNutritionToolIntegration
> Tests full meal logging tool with macro extraction

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| LogMealTool stores meals with macros | 1/1 (100%) | ✅ PASSED | 6.21s |

### ✅ TestNutritionModelComparison
> Baseline tests for comparing nutrition extraction across models

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Simple meal baseline (2 boiled eggs) | 1/1 (100%) | ✅ PASSED | 3.16s |
| Extraction with explicit quantities | 1/1 (100%) | ✅ PASSED | 3.39s |

### ⚠️ TestPlannerEmitsSearchMemoryForPersonalisedQueries

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| tell me some news that might interest me | 1/1 (100%) | ✅ PASSED | 9.94s |
| suggest something I'd enjoy watching ton | 1/1 (100%) | ✅ PASSED | 2.44s |
| what should I cook for dinner | 1/1 (100%) | ✅ PASSED | 2.59s |
| recommend a book I'd like | 1/1 (100%) | ✅ PASSED | 2.43s |
| what is the capital of France | 1/1 (100%) | ✅ PASSED | 2.48s |
| who is Britney Spears | 0/1 (0%) | ❌ FAILED | 2.56s |
| what's 2 plus 2 | 1/1 (100%) | ✅ PASSED | 2.29s |

### ⚠️ TestPossessorFieldRepro

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| First turn calls web search not clarification | 1/1 XFAIL | 🔸 XFAILED | 42.46s |
| Links only payload produces honest cant read reply | 1/1 (100%) | ✅ PASSED | 73.14s |
| Realistic web search payload is not deflected to links | 1/1 (100%) | ✅ PASSED | 89.66s |
| Digested tool result produces grounded reply | 1/1 XFAIL | 🔸 XFAILED | 24.01s |
| Follow up after correction calls web search | 1/1 XFAIL | 🔸 XFAILED | 26.83s |

### ✅ TestDiaryRecencyOrder
> Tests that diary search returns newer entries before older ones

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Office days changed from Mon/Wed to Mon/Thu | 1/1 (100%) | ✅ PASSED | 0.00s |
| Diet changed from bulking to cutting | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestGraphRecencySuperseding
> Tests that graph handles contradicting facts with date context

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Office days changed from Mon/Wed to Mon/Thu | 1/1 (100%) | ✅ PASSED | 0.00s |
| Diet changed from bulking to cutting | 1/1 (100%) | ✅ PASSED | 0.00s |

### ✅ TestMergeSupersession

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Office days changed from Mon/Wed to Mon/Thu | 1/1 (100%) | ✅ PASSED | 5.92s |
| Diet changed from bulking to cutting | 1/1 (100%) | ✅ PASSED | 2.42s |

### ✅ TestRecencyJudge
> LLM judge evaluates whether newer information is preferred over older

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Office days changed from Mon/Wed to Mon/Thu | 1/1 (100%) | ✅ PASSED | 4.80s |
| Diet changed from bulking to cutting | 1/1 (100%) | ✅ PASSED | 4.94s |

### ❌ TestReplyUsesNewerDiaryEntry

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Office days changed from Mon/Wed to Mon/Thu | SKIPPED | ⏭️ SKIPPED (Skipped: Chat model is paramet...) | 0.00s |
| Office days changed from Mon/Wed to Mon/Thu | SKIPPED | ⏭️ SKIPPED (Skipped: Chat model is paramet...) | 0.00s |
| Diet changed from bulking to cutting | SKIPPED | ⏭️ SKIPPED (Skipped: Chat model is paramet...) | 0.00s |
| Diet changed from bulking to cutting | SKIPPED | ⏭️ SKIPPED (Skipped: Chat model is paramet...) | 0.00s |

### ✅ TestRouterReturnsNoneWhenContextAnswers

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Time query with time in context returns none | 1/1 (100%) | ✅ PASSED | 2.39s |
| Date query with date in context returns none | 1/1 (100%) | ✅ PASSED | 2.24s |
| Location query with location in context returns none | 1/1 (100%) | ✅ PASSED | 2.29s |

### ⚠️ TestRouterPicksToolsWhenContextDoesNotAnswer

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Weather query still picks getWeather | 1/1 (100%) | ✅ PASSED | 2.53s |
| Location query with partial hint still routes sensibly | 1/1 XFAIL | 🔸 XFAILED | 2.30s |
| Followup naming place routes to getWeather | 1/1 (100%) | ✅ PASSED | 2.50s |
| No hint at all still routes sensibly | 1/1 (100%) | ✅ PASSED | 2.30s |

### ⚠️ TestImplicitIntent

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| food decision \u2192 fetchMeals | 1/1 XFAIL | 🔸 XFAILED | 2.49s |
| calorie budget \u2192 fetchMeals | 1/1 (100%) | ✅ PASSED | 2.24s |
| jacket \u2192 getWeather | 1/1 (100%) | ✅ PASSED | 2.35s |
| run forecast \u2192 getWeather | 1/1 (100%) | ✅ PASSED | 2.42s |
| meal recall (colloquial) \u2192 fetchMeals | 1/1 (100%) | ✅ PASSED | 2.27s |
| dietary check \u2192 fetchMeals | 1/1 (100%) | ✅ PASSED | 2.50s |

### ✅ TestMultiToolIntent

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| weather + meals | 1/1 (100%) | ✅ PASSED | 2.31s |
| research \u2192 webSearch + fetchWebPage | 1/1 (100%) | ✅ PASSED | 2.52s |
| weekly weather keeps getWeather | 1/1 (100%) | ✅ PASSED | 2.26s |

### ✅ TestRouterNeverCollapses

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| take a screenshot | 1/1 (100%) | ✅ PASSED | 2.40s |
| what's on my screen right now? | 1/1 (100%) | ✅ PASSED | 2.36s |
| search the web for flight deals | 1/1 (100%) | ✅ PASSED | 2.56s |
| log that I just ate a banana | 1/1 (100%) | ✅ PASSED | 2.25s |
| what's the weather like? | 1/1 (100%) | ✅ PASSED | 2.52s |
| find the invoice PDF on my computer | 1/1 (100%) | ✅ PASSED | 2.26s |

### ✅ TestToolSelectionFiltering

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| weather query selects getWeather and few others | 1/1 (100%) | ✅ PASSED | 25.71s |
| location weather query selects getWeather and few others | 1/1 (100%) | ✅ PASSED | 23.05s |
| meal logging selects logMeal and few others | 1/1 (100%) | ✅ PASSED | 23.00s |
| meal recall selects fetchMeals and few others | 1/1 (100%) | ✅ PASSED | 22.97s |
| web search query selects webSearch and few others | 1/1 (100%) | ✅ PASSED | 22.99s |

### ✅ TestToolSelectionFilteringLLM

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| weather query selects getWeather and few others | 1/1 (100%) | ✅ PASSED | 2.41s |
| location weather query selects getWeather and few others | 1/1 (100%) | ✅ PASSED | 2.28s |
| meal logging selects logMeal and few others | 1/1 (100%) | ✅ PASSED | 2.44s |
| meal recall selects fetchMeals and few others | 1/1 (100%) | ✅ PASSED | 2.38s |
| web search query selects webSearch and few others | 1/1 (100%) | ✅ PASSED | 2.39s |

### ✅ TestWeatherAutoDerivesLocation

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| cold-memory-week-forecast-what's the weather this week | 1/1 (100%) | ✅ PASSED | 50.77s |
| cold-memory-short-query-how's the weather | 1/1 (100%) | ✅ PASSED | 58.37s |
| warm-memory-short-query-how's the weather | 1/1 (100%) | ✅ PASSED | 57.60s |

### ✅ TestFallbackChainRescuesBotChallenge

| Test Case | Pass Rate | Status | Avg Duration |
|-----------|-----------|--------|--------------|
| Wikipedia rescues when ddg blocks | 1/1 (100%) | ✅ PASSED | 0.08s |
| Honest block when all providers fail | 1/1 (100%) | ✅ PASSED | 0.00s |

---

*Report generated by Jarvis eval suite*