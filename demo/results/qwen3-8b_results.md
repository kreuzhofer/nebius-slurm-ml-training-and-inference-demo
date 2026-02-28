# Evaluation Results (100 test examples)

| | Model | Accuracy | Correct |
|---|---|---|---|
| Base | `/mnt/data/demo/models/Qwen3-8B` | 2.0% | 2/100 |
| Fine-tuned | `/mnt/data/demo/output/qwen3-8b-sql` | 88.0% | 88/100 |

**Improvement: +86.0%**

## Examples where fine-tuning fixed the output

### Example 1

**Question:** When Essendon played away; where did they play?

**Ground truth:** `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

**Base model:** `SELECT venue FROM table_name_50 WHERE away_team = 'Essendon'`

**Fine-tuned:** `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

### Example 2

**Question:** What is the lowest numbered game against Phoenix with a record of 29-17?

**Ground truth:** `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

**Base model:** `SELECT MIN(game) FROM table_name_61 WHERE opponent = 'Phoenix' AND record = '29-17';`

**Fine-tuned:** `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

### Example 5

**Question:** What is the name of the player who is Sco and moving to greenock morton in the summer?

**Ground truth:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

**Base model:** `SELECT name FROM table_name_83 WHERE name = 'Sco' AND moving_to = 'greenock morton' AND transfer_window = 'summer'`

**Fine-tuned:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

### Example 6

**Question:** Of all the contestants who got voted, what is the contestant number and name of the contestant who got least votes?

**Ground truth:** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

**Base model:** `SELECT v.contestant_number, c.contestant_name
FROM votes v
JOIN contestants c ON v.contestant_number = c.contestant_number
GROUP BY v.contestant_number, c.contestant_name
ORDER BY COUNT(*) ASC
LIMIT 1;`

**Fine-tuned:** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

### Example 7

**Question:** Which venue had the result 7-1?

**Ground truth:** `SELECT venue FROM table_name_50 WHERE result = "7-1"`

**Base model:** `SELECT venue FROM table_name_50 WHERE result = '7-1';`

**Fine-tuned:** `SELECT venue FROM table_name_50 WHERE result = "7-1"`

