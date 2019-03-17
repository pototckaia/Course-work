import psycopg2 as psy
from psycopg2 import sql
import os
from operator import itemgetter

conn = psy.connect(dbname='mok', user='postgres', 
					password='', host='localhost')

def copy_cvs_predictant_storage(point_id, type_id):
	with conn.cursor() as cursor:
		path = os.path.dirname(os.path.abspath(__file__)) + \
			   os.sep + 'date/{}_{}.csv'.format(point_id, type_id)

		columns = ('actual_date', 'value')
		from_table = 'predictant_storage'

		query = sql.SQL("""SELECT {} FROM {} 
						WHERE \"point_id\" = {} and \"predictant_id\" = {}
						""").format(
			sql.SQL(',').join(map(sql.Identifier, columns)),
			sql.Identifier(from_table),
			sql.Literal(point_id),
			sql.Literal(type_id)
		).as_string(conn)
		print(query)
		outputquery = "COPY ({}) TO STDOUT WITH CSV HEADER;".format(query)

		with open(path, 'w+') as f:
		    cursor.copy_expert(outputquery, f)


def create_new_db_predictor_storage():
	query_map_character = """select \"id\", \"name\" 
						  from \"predictor\"
						  where not ('[1, 100]'::int4range @> id) and 
						        not ('[155, 158]'::int4range @> id)  and 
						        not ('[255, 258]'::int4range @> id) and 
						        not ('[355, 358]'::int4range @> id)"""
	with conn.cursor() as cursor:
		cursor.execute(query_map_character)
		columns = list(cursor.fetchall())
		name_columns = map(itemgetter(1), columns)
		table_name = 'predictor_storage_union_character'

		query = sql.SQL("""CREATE TABLE if not exists {}(
							"point_id" 	  int4,
		 	 				"actual_date" timestamp,
		 					"offset_h" 	int4,
		 	 				{}) 
		 				""").format(
		 	sql.Identifier(table_name), 
		 	sql.SQL(' float4,\n').join(map(sql.Identifier, name_columns)) + sql.SQL(' float4'),
		)
		cursor.execute(query)
		conn.commit()

		query = sql.SQL("""insert into {0}({1})
					select DISTINCT on ("point_id", "actual_date") {1} 
				    from "predictor_storage"
				    order by {1} DESC
				    """).format(
				    	sql.Identifier(table_name), 
				    	sql.SQL(',').join(map(sql.Identifier, ('point_id', 'actual_date', 'offset_h'))),
				    )
		cursor.execute(query)
		conn.commit()

		for col in columns:
			query = sql.SQL("""
					   update predictor_storage_union_character as new_p
					   set ({}) = (select old_p.value
					   			from predictor_storage as old_p
					   			where old_p.point_id = new_p.point_id and 
					   			      old_p.actual_date  = new_p.actual_date and
					   			      old_p.offset_h = new_p.offset_h and 
					                  old_p.predictor_id = %s
					            );
					  	""").format(sql.Identifier(col[1]))
			cursor.execute(query, (col[0], ))
			conn.commit()

def copy_new_db_predictor_id(point_id):
	with conn.cursor() as cursor:
		path = os.path.dirname(os.path.abspath(__file__)) + \
			   os.sep + 'date/character_{}.csv'.format(point_id)

		from_table = 'predictor_storage_union_character'

		query = sql.SQL("""SELECT * FROM {} 
						WHERE \"point_id\" = {}
						""").format(
			sql.Identifier(from_table),
			sql.Literal(point_id),
		).as_string(conn)
		outputquery = "COPY ({}) TO STDOUT WITH CSV HEADER;".format(query)

		with open(path, 'w+') as f:
		    cursor.copy_expert(outputquery, f)

# query_type = 'select "id" from "predictants"'
# query_station = 'select "id" from "points"'
#predictor_storage
#point_id, predictor_id, actual_date, offset_h, value


copy_cvs_predictant_storage(31088, 103)
# copy_cvs_predictor_storage()
# copy_new_db_predictor_id(31088)