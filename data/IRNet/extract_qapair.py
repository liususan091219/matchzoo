from utils import load_dataset, extract_pair

if __name__ == '__main__':
    dataset_dir = "../../../nl2sql/code/IRNet_data/data/"
    table_path = "../../../nl2sql/code/IRNet_data/"

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas = load_dataset(dataset_dir, table_path)

    fout2 = open("../../MatchZoo_data/WikiQA/corpus.txt", "w")

    extract_pair(sql_data, "train", fout2)
    extract_pair(val_sql_data, "dev", fout2)

    fout2.close()