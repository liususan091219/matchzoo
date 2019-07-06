import os, json
import re, sys, nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

regex = re.compile("^(?P<action>[a-zA-Z0-9]*)\((?P<colid>\d+)\)$")

reload(sys)
sys.setdefaultencoding('utf8')

stopwords = set(nltk.corpus.stopwords.words('english'))

def remove_stopwords(g_str):
    str_tokens = g_str.lower().split()
    str_conc = ""
    for each_str in str_tokens:
        if each_str not in stopwords:
            str_conc += each_str + " "
    return str_conc.rstrip()

def lemmatize_str(g_str):
    str_tokens = g_str.split()
    str_conc = ""
    for each_str in str_tokens:
            str_conc += wordnet_lemmatizer.lemmatize(each_str.lower()) + " "
    return str_conc.rstrip()

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
    answer_int = 0

    fout = open("../../../MatchZoo_data/irnet/relation_" + train_dev + ".txt", "w")

    for i in range(0, len(sql_data)):

        question_id = "Q" + str(i)

        this_data = sql_data[i]
        db_id = this_data["db_id"]

        this_src = remove_stopwords(this_data["question"])
        this_src = lemmatize_str(this_src)
        this_colset = this_data["col_set"]

        this_pos_labels = extract_col(this_data["rule_label"])

        fout2.write(question_id + " " + this_src + "\n")

        for j in range(0, len(this_colset)):

            answer_id = "D" + str(answer_int)
            answer_int += 1

            if j in this_pos_labels:
                this_label = 1
            else:
                this_label = 0

            fout.write(str(this_label) + "\t" + question_id + "\t" + answer_id + "\n")

            this_colstr = this_colset[j]

            fout2.write(answer_id + "\t" + this_colstr + "\t" + db_id + "\n")

    fout.close()

def process(sql_data, table_data, val_data):
    output_tab = {}
    tables = {}
    tabel_name = set()
    # remove_list = ['?', '.', ',', "''", '``', '(', ')', "'"]
    remove_list = list()

    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['schema_len'] = []
        length = {}
        for col_tup in temp['col_map']:
            length[col_tup[0]] = length.get(col_tup[0], 0) + 1
        for l_id in range(len(length)):
            temp['schema_len'].append(length[l_id-1])
        temp['foreign_keys'] = table['foreign_keys']
        temp['primary_keys'] = table['primary_keys']
        temp['table_names'] = table['table_names']
        temp['column_types'] = table['column_types']
        db_name = table['db_id']
        tabel_name.add(db_name)
        # print table
        output_tab[db_name] = temp
        tables[db_name] = table
    # print tabel_name
    # quit()
    output_sql = []
    for i in range(len(sql_data)):
        sql = sql_data[i]
        sql_temp = {}

        # add query metadata
        for key, value in sql.items():
            sql_temp[key] = value
        sql_temp['question'] = sql['question']

        sql_temp['question_tok'] = [wordnet_lemmatizer.lemmatize(x).lower() for x in sql['question_toks'] if x not in remove_list]
        # for vo_idx in range(len(sql_temp['question_tok'])):
        #     if sql_temp['question_tok'][vo_idx] in filter_voc:
        #         sql_temp['question_tok'][vo_idx] = 'UNK'
        # rule_label is string with value like "Sel(2) AGG(1-max) COL(5-budget in billions) AGG(2-min) COL(5-budget in billions) SUP(none) FIL(end) ORD(end)"
        sql_temp['rule_label'] = sql['rule_label']
        if len(sql["col_set"]) >= 155:
            continue
        sql_temp['col_set'] = sql['col_set']
        sql_temp['query'] = sql['query']
        # dre_file.write(sql['query'] + '\n')
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']
        table = tables[sql['db_id']]
        val = val_data[sql['db_id']]

        sql_temp['col_org'] = table['column_names_original']
        sql_temp['table_org'] = table['table_names_original']
        sql_temp['table_names'] = table['table_names']
        sql_temp['fk_info'] = table['foreign_keys']
        tab_cols = [col[1] for col in table['column_names']]
        sql_temp["col_val"] = val

        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
        sql_temp['col_iter'] = col_iter
        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1]) #GOLD for sel and agg

        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])
                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond) #GOLD for COND [[col, op],[]]

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']] #assume only one groupby
        having_cond = []
        if len(sql['sql']['having']) > 0:
            gt_having = sql['sql']['having'][0] # currently only do first having condition
            having_cond.append([gt_having[2][1][0]]) # aggregator
            having_cond.append([gt_having[2][1][1]]) # column
            having_cond.append([gt_having[1]]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        sql_temp['group'].append(having_cond) #GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # process order by / limit
        order_aggs = []
        order_cols = []
        sql_temp['order'] = []
        order_par = 4
        gt_order = sql['sql']['orderby']
        limit = sql['sql']['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]] # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        sql_temp['order'] = [order_aggs, order_cols, order_par] #GOLD for ORDER [[[agg], [col], [dat]], []]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect'] is not None:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        if 'stanford_tokenized' in sql:
            sql_temp['stanford_tokenized'] = sql['stanford_tokenized']
        if 'stanford_pos' in sql:
            sql_temp['stanford_pos'] = sql['stanford_pos']
        if 'stanford_dependencies' in sql:
            sql_temp['stanford_dependencies'] = sql['stanford_dependencies']
        if 'hardness' in sql:
            sql_temp['hardness'] = sql['hardness']
        if 'question_labels' in sql:
            sql_temp['question_labels'] = sql['question_labels']

        output_sql.append(sql_temp)
    return output_sql, output_tab

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

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