# 🧪 Jarvis Evaluation Report

**Generated:** 2026-04-27 (ad-hoc update on this branch only — see "Memory merge consolidation" below; rest of report inherited from develop's last full regen)

## 📊 TL;DR

**Overall:** 🟢 **330/348 passed (94.8%)** across all categories *(merge consolidation now also covers meta-narrative pruning; the previous xfail flipped to XPASS on the strengthened prompt)*

| Category | Model | Passed | Failed | Skipped | Pass Rate |
|----------|-------|-------:|-------:|--------:|----------:|
| 🤖 Agent behaviour | `gemma4:e2b` | 129 | 11 | 2 | 🟢 92.1% |
| 🤖 Agent behaviour | `gpt-oss:20b` | 145 | 7 | 0 | 🟢 95.4% |
| 🎤 Intent judge | `gemma4:e2b` (fixed) | 47 | 0 | 0 | 🟢 100.0% |
| 🧠 Memory merge consolidation | `gemma4:e2b` | 9 | 0 | 0 | 🟢 100.0% (1 xpass) |

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
| Active hot window follow up accepted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Adversarial: all three branches in one summary | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Adversarial: food preference (USER) vs list-length rule (DIRECTIVES) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent calls webSearch for info queries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent chains search → fetch for details | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent uses memory + nutrition data | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Assistant checks memory before asking about interests | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Assistant does not deny having long-term memory | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: deflection without attempting answer | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: empty acknowledgment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: generic greeting ignores query | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Casual statement without wake word rejected | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Chained research: who directed Possessor and what else have they made | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Correction loop accepts single or retry | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Cross turn pronoun resolution | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| DIRECTIVES: tone, length, forbidden phrases, address form | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Date query with date in context returns none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Diet changed from bulking to cutting | ⏭️ SKIPPED | 🔸 1/1 XFAIL |
| Digested tool result produces grounded reply | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Director-then-filmography needs two searches | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Enrichment results appear in system message | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Enrichment skips questions answered by context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Escape hatch then follow up action | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Evaluator emits structured tool call for obvious search | ✅ 1/1 (100%) | 🔸 1/1 XFAIL |
| Extraction with explicit quantities | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| First turn calls web search not clarification | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Follow up after correction calls web search | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Follow up resolves pronoun in search query | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Follow-up references previous turn context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Followup naming place routes to getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: brief but informative | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: complete weekly forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Graph-enriched facts surface in the reply, no denial | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Greeting: hello | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Greeting: ni hao (Chinese) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Handles ambiguous portion descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Honest block when all providers fail | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Hot window query is directed and non empty | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Identity query does not trigger recommendation engagement rule | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Identity query surfaces multiple user facts when present | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Identity query surfaces user stated fact over past qa | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Identity query with only past qa returns none or no false facts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Instruction: be more brief | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Instruction: use Celsius | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Judge echo claim overridden in hot window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| LLM uses enrichment-surfaced interests for personalised search | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Links only payload produces honest cant read reply | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Location context flows to search queries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Location query with location in context returns none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Location query with partial hint still routes sensibly | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| LogMealTool stores meals with macros | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Max-turn cap delivers a digest reply, never silence | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: personalized news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: time-based recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: topic recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Mixed summary: keep novel facts, drop stale weather/recommendations | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Navigate prose gets nudged into tool call | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| No deflection: tech news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: time query | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| No deflection: tomorrow weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: weekly rain forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No email tool declines honestly | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No hint at all still routes sensibly | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No wake word rejected despite judge | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: local business details and user location | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: non-English summary (Turkish) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: relocation plans and employment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: user diet plan and preferred recipe | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nudge cap stops loop | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: cheeseburger with fries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: chicken with broccoli | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: oatmeal with banana | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Office days changed from Mon/Wed to Mon/Thu | ⏭️ SKIPPED | 🔸 1/1 XFAIL |
| Omits deflection narration for unknown entity | ✅ 1/1 (100%) | 🔸 1/1 XFAIL |
| Omits deflection when topic never resolved | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Open-ended prompt grounds in stored knowledge | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Parallel weather lookup: compare Paris and London | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Preserves legitimate user preferences | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Realistic web search payload is not deflected to links | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Recommendation query still surfaces engagement when user facts present | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reframing: life events framed as facts with temporal context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reframing: requests become knowledge, not interaction descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: assistant self-references (recommendations are not knowledge) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: stale temporal snapshots (weather, time of day) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Restaurant recommendation surfaces past cuisine interest | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns NONE for non-food inputs | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns valid JSON with all required fields | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Simple meal baseline (2 boiled eggs) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Single weather query ends after one tool call | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| Speech long after tts requires wake word | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Stop during tts interrupts immediately | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Time query with time in context returns none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool calls literal not surfaced after web search | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: explicit tool mention | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: vague go ahead | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: vague just try | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Toolsearchtool widens then navigate | 🔸 1/1 XFAIL | 🔸 1/1 XFAIL |
| Topic switch: search → weather uses getWeather | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Topic switch: weather → store hours uses webSearch | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Trivial conversations produce no extracted facts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tts echo segments skipped user query extracted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Turn1 possessor then turn2 weather | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Two-turn celebrity flow: identity then pronoun follow-up | 🔸 1/1 XFAIL | ❌ 0/1 (0%) |
| USER: identity, location, pets, diet, job | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Unknown entity with poisoned diary still triggers web search live | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Unknown entity: Piranesi (book) | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Unknown entity: Possessor (film) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Unknown entity: have-you-heard-of (Piranesi) | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Unknown entity: permission-framed (Possessor) | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Unrelated domain still returns none | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Unrelated topics are not welded into one clause | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| User query not confused with echo after tts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Utterance started during tts treated as hot window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| WORLD: local business details, film attribution | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Wake word query after echo segments | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Wake word query uses judge extraction | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Watch recommendation surfaces recently discussed films | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Weather query is answered with current conditions | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Weather query still picks getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Weather query still triggers tools after a greeting | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Wikipedia payload produces grounded reply | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Wikipedia rescues when ddg blocks | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| calorie budget \u2192 fetchMeals | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| cold-memory-short-query-how's the weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| cold-memory-week-forecast-what's the weather this week | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| dietary check \u2192 fetchMeals | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| explicit-recall-then-search | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| find the invoice PDF on my computer | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| food decision \u2192 fetchMeals | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| jacket \u2192 getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| location weather query selects getWeather and few others | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| log that I just ate a banana | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| meal logging selects logMeal and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| meal recall (colloquial) \u2192 fetchMeals | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| meal recall selects fetchMeals and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| news-interesting-for-me | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| news-of-interest-to-me | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| news-that-would-interest-me | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| recommend a book I'd like | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| research \u2192 webSearch + fetchWebPage | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| run forecast \u2192 getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| search the web for flight deals | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| suggest something I'd enjoy watching ton | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| take a screenshot | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| tell me some news that might interest me | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| warm-memory-short-query-how's the weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| weather + meals | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| weather query selects getWeather and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| web search query selects webSearch and few others | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| weekly weather keeps getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| what is the capital of France | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| what should I cook for dinner | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| what's 2 plus 2 | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| what's on my screen right now? | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| what's the weather like? | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| who is Britney Spears | ❌ 0/1 (0%) | ✅ 1/1 (100%) |

---

## 🎤 Intent judge

> Pinned to `gemma4:e2b` (the voice intent classifier). Not affected by the judge model.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| Hot window mode indicated in prompt | 1/1 (100%) | ✅ |
| Old query not re extracted | 1/1 (100%) | ✅ |
| Processed segment not reextracted | 1/1 (100%) | ✅ |
| Returns none when ollama unavailable | 1/1 (100%) | ✅ |
| System prompt has echo guidance | 1/1 (100%) | ✅ |
| Tts text included for echo detection | 1/1 (100%) | ✅ |
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
| cross_segment_go_ahead_and_answer | 1/1 (100%) | ✅ |
| cross_segment_hot_window_followup | 1/1 (100%) | ✅ |
| cross_segment_imperative_superseded_by_new_question | 1/1 (100%) | ✅ |
| echo_plus_followup_extracted | 1/1 (100%) | ✅ |
| echo_plus_rejected_similar_plus_wake_retry | 1/1 (100%) | ✅ |
| hot_window_override_topicless_followup | 1/1 (100%) | ✅ |
| hot_window_simple_followup | 1/1 (100%) | ✅ |
| mentioned_in_narrative_past_tense | 1/1 (100%) | ✅ |
| multi_person_vague_reference | 1/1 (100%) | ✅ |
| multi_person_weather_discussion | 1/1 (100%) | ✅ |
| multiple_echoes_then_interrupt | 1/1 (100%) | ✅ |
| no_wake_word_casual_speech | 1/1 (100%) | ✅ |
| no_wake_word_in_buffer | 1/1 (100%) | ✅ |
| stop_command_during_tts | 1/1 (100%) | ✅ |
| user_followup_statement_after_question_nihilism | 1/1 (100%) | ✅ |
| wake_word_after_narrative_addresses_assistant | 1/1 (100%) | ✅ |
| wake_word_command_timer | 1/1 (100%) | ✅ |
| wake_word_mid_sentence | 1/1 (100%) | ✅ |
| wake_word_open_imperative_give_me_advice | 1/1 (100%) | ✅ |
| wake_word_open_imperative_say_something | 1/1 (100%) | ✅ |
| wake_word_open_imperative_surprise_me | 1/1 (100%) | ✅ |
| wake_word_open_imperative_tell_me_a_joke | 1/1 (100%) | ✅ |
| wake_word_open_imperative_tell_me_anything | 1/1 (100%) | ✅ |
| wake_word_share_statement_burger | 1/1 (100%) | ✅ |
| wake_word_share_statement_feeling | 1/1 (100%) | ✅ |
| wake_word_share_statement_trailing | 1/1 (100%) | ✅ |
| wake_word_simple_question | 1/1 (100%) | ✅ |
| wake_word_statement_remember | 1/1 (100%) | ✅ |
| wake_word_trailing_after_named_entity | 1/1 (100%) | ✅ |

---

## 🧠 Memory merge consolidation

> Exercises `merge_node_data` against a real picker model. Pins the rewrite-on-write merge against its five advertised behaviours: dedupe of near-duplicates, pattern consolidation of repeated activities, independence (unrelated facts coexist, no silent erasure), meta-narrative pruning (assistant-narrating extractor leftovers get scrubbed), and end-to-end correctness of the batched signature. Run via `pytest evals/test_merge_consolidation.py`.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| Dedupe — same fact, different wording (lives-in vs based-in London) | 1/1 (100%) | ✅ |
| Dedupe — job title rephrased | 1/1 (100%) | ✅ |
| Pattern — repeated sushi meals fold into "regularly eats sushi" | 1/1 (100%) | ✅ |
| Pattern boundary — distinct one-off dated events stay distinct | 1/1 (100%) | 🎉 |
| Independence — peanut allergy + tea preference survive unrelated hiking fact | 1/1 (100%) | ✅ |
| Independence — software-engineer job survives unrelated guitar fact | 1/1 (100%) | ✅ |
| Meta-narrative — capability-denial line dropped, real directive kept | 1/1 (100%) | ✅ |
| Meta-narrative — assistant-suggested line dropped, factual lookup survives | 1/1 (100%) | ✅ |
| Meta-narrative — clean directives node not over-pruned | 1/1 (100%) | ✅ |
| Batched merge — three independent new facts in one call all land | 1/1 (100%) | ✅ |

**Notes:** the pattern-boundary case is still marked `xfail(strict=False)` in source — the upstream small-model regression it captured is not formally fixed, but the strengthened META-NARRATIVE rule appears to have indirectly tightened the picker's general independence handling and the case now XPASSes (3/3 reps on `gemma4:e2b`). Left as xfail until a more targeted fix lands so the marker still surfaces a regression if the side-effect goes away.

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