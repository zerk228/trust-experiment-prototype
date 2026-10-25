# Experiment protocol (Assignment 3, Block C)

Design: 2x2 between-subjects. Difficulty (simple / complex) x accuracy (correct / incorrect). Conditions: A simple+correct, B simple+incorrect, C complex+correct, D complex+incorrect. 12 per condition planned.

## Order and timing (same for every participant)
1. Welcome, instruction read aloud word for word, oral agreement (1 min)
2. Prior chatbot use question, 1-5 (1 min)
3. Chat task: three situations, each reply appears after a fixed 2 second delay (about 4 min)
4. Trust questionnaire (10 questions), realism item, two check questions, one optional sentence (about 6 min)
5. Debrief: replies were pre-written, some wrong on purpose, no real order; ask what the participant thought was tested; type the answer and save (1 min)
Total: about 13 min.

## Instruction (read word for word)
"Thank you for taking part. Imagine that you ordered two items from an online store called ShopNest a few days ago, and you want to find out about your order. You will chat with ShopNest's AI support assistant. You will see three situations, one at a time. For each one, send the message shown on the screen to the assistant, then read its reply. Nothing you do is right or wrong. After the three exchanges you will answer a short questionnaire about the assistant. Taking part is voluntary, you can stop at any time, and your answers are anonymous. Do you have any questions?"

## Checklist before every session
- [ ] Same laptop and browser as in earlier sessions
- [ ] Prototype opened on the researcher setup screen
- [ ] Participant ID entered (next free number)
- [ ] Condition left on "Auto (next in block)" (manual only with a reason in the log)
- [ ] "Pilot session" box ticked only for pilot sessions
- [ ] Instruction read word for word
- [ ] Participant does not see the setup screen
- [ ] After the session: row saved, one line added to the research log
- [ ] After each session day: CSV exported and uploaded to data/raw

## Fixed rules
- Reply delay 2 seconds; agent called "AI assistant" in all conditions
- Each incorrect reply within 15 percent of the words of its correct reply (see scenarios/scenario_pool.csv)
- Scoring key fixed before data collection (TRUST_ITEMS in index.html)
- Success criterion: trust total 8 points higher after correct replies, and a drop at least 4 points larger for complex questions
