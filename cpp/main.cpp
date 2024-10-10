#include <arrow/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

int
main (int argc, char **argv)
{
  if (argc != 2)
    {
      fprintf (stderr, "Usage: %s <parquet_file>\n", argv[0]);
      return 1;
    }

  // Read Parquet file
  // std::shared_ptr<arrow::Table> table = read_parquet_file (argv[1]);

  // Extract data and feed to Flex
  // extract_data_for_flex (table);

  // Parse and generate tasks
  // yyparse ();

  return 0;
}
