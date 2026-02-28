# Evaluation Results (100 test examples)

| | Model | Accuracy | Correct |
|---|---|---|---|
| Base | `/mnt/data/demo/models/Qwen3-32B` | 3.0% | 3/100 |
| Fine-tuned | `/mnt/data/demo/output/qwen3-32b-sql` | 84.0% | 84/100 |

**Improvement: +81.0%**

## Examples where fine-tuning fixed the output

### Example 1

**Question:** When Essendon played away; where did they play?

**Ground truth:** `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

**Base model:** `SELECT venue FROM table_name_50 WHERE away_team = 'Essendon';`

**Fine-tuned:** `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

### Example 2

**Question:** What is the lowest numbered game against Phoenix with a record of 29-17?

**Ground truth:** `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

**Base model:** `SELECT game
FROM table_name_61
WHERE opponent = 'Phoenix' AND record = '29-17'
ORDER BY game ASC
LIMIT 1;`

**Fine-tuned:** `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

### Example 5

**Question:** What is the name of the player who is Sco and moving to greenock morton in the summer?

**Ground truth:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

**Base model:** `SELECT name FROM table_name_83 WHERE name = 'Sco' AND moving_to = 'greenock morton' AND transfer_window = 'summer';`

**Fine-tuned:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

### Example 7

**Question:** Which venue had the result 7-1?

**Ground truth:** `SELECT venue FROM table_name_50 WHERE result = "7-1"`

**Base model:** `SELECT venue FROM table_name_50 WHERE result = '7-1';`

**Fine-tuned:** `SELECT venue FROM table_name_50 WHERE result = "7-1"`

### Example 8

**Question:** What did the tournament that got an A in 1945 get in 1949?

**Ground truth:** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

**Base model:** `SELECT * FROM table_name_22 WHERE Id = 'A';`

**Fine-tuned:** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

