import sys
import subprocess


replacements = {' NOT NULL':''}

# Command with shell expansion
def gen_table(csvfile, sch_name, tb_name, sql_file):
    return("head -n 1000 '{0}' | tr [:upper:] [:lower:] | tr ' ' '_' | sed 's/#/num/' | csvsql -i mysql --db-schema {1} --tables {2} > {3}".format(csvfile, sch_name, tb_name, sql_file))

def gen_schema(sch_name, tb_name):
    return('''CREATE SCHEMA IF NOT EXISTS {0};
    DROP TABLE IF EXISTS {0}.{1};'''.format(sch_name, tb_name))

def clean_sql(fin, fout, csvfile, sch_name, tb_name):
    sch_sql = gen_schema(sch_name, tb_name)
    with open(fin) as infile, open(fout, 'w') as outfile:
        outfile.write(sch_sql)
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)
        endline = """LOAD DATA LOCAL INFILE '{0}'
                    INTO TABLE {1}.{2}
                    FIELDS TERMINATED BY ','
                    ENCLOSED BY '"'
                    LINES TERMINATED BY "\n"
                    IGNORE 1 ROWS;""".format(csvfile, sch_name, tb_name)
        outfile.write(endline)


if __name__ == "__main__":
    # 1 : sql text file in to be cleaned
    # 2 : sql text cleaned file out (sql script produced)
    # 3 : csv file name
    # 4 : schema
    # 5 : table name
    # call: python etl.py load.sql load_clean.sql 1909.csv mobi_data trips
    #head -n 1000 '1909.csv' | tr [:upper:] [:lower:] | tr ' ' '_' | sed 's/#/num/' | csvsql -i mysql --db-schema mobi_data --tables apcfall2019 > apc_table.sql
    shell1 = gen_table(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[1])
    subprocess.call(shell1, shell=True)
    clean_sql(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    subprocess.call('mysql -h localhost -u root -p --local-infile < {0}'.format(sys.argv[2]), shell=True)
