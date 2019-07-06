import os, json
import re

regex = re.compile("^(?P<action>[a-zA-Z0-9]*)\((?P<colid>\d+)\)$")

def extract_col(g_str):
        tokens = g_str.split()
        colid_set = set()
        for i in range(0, len(tokens)):
                result = regex.search(tokens[i])
                colid = int(result.group("colid"))
                action = result.group("action")
                if action == "C":
                   colid_set.add(colid)
        return colid_set

def extract_pair(sql_data, train_dev, fout2):
    answer_id = 0

    fout = open("../../../MatchZoo_data/WikiQA/relation_" + train_dev + ".txt", "w")

    for i in range(0, len(sql_data)):

        question_id = "Q" + str(i)

        this_data = sql_data[i]
        db_id = this_data["db_id"]

        this_src = this_data["question"]
        this_colset = this_data["col_set"]

        this_pos_labels = extract_col(this_data["rule_label"])

        fout2.write(question_id + " " + this_src + "\n")

        for j in range(0, len(this_colset)):

            answer_id = "D" + str(answer_id)
            answer_id += 1

            if j in this_pos_labels:
                this_label = 1
            else:
                this_label = 0

            fout.write(str(this_label) + "\t" + question_id + "\t" + answer_id + "\n")

            this_colstr = this_colset[i]

            fout2.write(answer_id + "\t" + this_colstr + "\t" + db_id + "\n")

    fout.close()

def load_data_new(sql_path, table_data, val_data):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    sql_data_new, table_data_new = process(sql_data, table_data, val_data)  # comment out if not on full dataset

    schemas = {}
    for tab in table_data:
        schemas[tab['db_id']] = tab

    return sql_data_new, table_data_new, schemas


def load_dataset(dataset_dir, table_path):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(table_path, "tables_mapped.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train_mapped.json")
    DEV_PATH = os.path.join(dataset_dir, "dev_mapped.json")
    TEST_PATH = os.path.join(dataset_dir, "dev_mapped.json")

    VAL_PATH = os.path.join(table_path, "db2topvals.json")

    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)

    with open(VAL_PATH) as inf:
        val_data = json.load(inf)

    train_sql_data, train_table_data, schemas_all = load_data_new(TRAIN_PATH, table_data, val_data)
    dev_sql_data, dev_table_data, schemas = load_data_new(DEV_PATH, table_data, val_data)
    test_sql_data, test_table_data, schemas = load_data_new(TEST_PATH, table_data, val_data)

    return train_sql_data, train_table_data, dev_sql_data, dev_table_data,\
            test_sql_data, test_table_data, schemas_all