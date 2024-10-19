import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CodeAnalysisTasks:
    """
    A class for generating and saving code analysis tasks from a given DataFrame.

    This class processes a DataFrame containing code snippets, docstrings, and summaries,
    and generates four types of code analysis tasks: Function Explanation, Parameter Explanation,
    Summary, and Docstring Completeness.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing code data.
        tasks (List[Dict[str, str]]): A list to store generated tasks.
        max_rows (Optional[int]): Maximum number of rows to process from the DataFrame.
    """

    def __init__(self, df: pd.DataFrame, max_rows: Optional[int] = None):
        """
        Initialize the CodeAnalysisTasks object.

        Args:
            df (pd.DataFrame): The input DataFrame containing code data.
            max_rows (Optional[int]): Maximum number of rows to process from the DataFrame.
                If None, all rows will be processed.
        """
        self.df = df if max_rows is None else df.head(max_rows)
        self.tasks: List[Dict[str, str]] = []

    def split_docstring(self, docstring: str) -> Tuple[str, str]:
        """
        Split a docstring into summary and details parts.

        Args:
            docstring (str): The full docstring to split.

        Returns:
            Tuple[str, str]: A tuple containing the summary and details parts of the docstring.
        """
        parts = re.split(r"\n\s*(Args|Parameters|Returns|Raises):", docstring, 1)
        if len(parts) > 1:
            return parts[0].strip(), parts[1].strip()
        else:
            return docstring.strip(), ""

    def generate_tasks(self) -> None:
        """
        Generate tasks from the input DataFrame.

        This method processes each row in the DataFrame and generates four types of tasks:
        Function Explanation, Parameter Explanation, Summary, and Docstring Completeness.
        """
        for _, row in self.df.iterrows():
            code = row["code"]
            docstring = row["docstring"]
            summary = row["summary"]

            # Task 1: Function Explanation
            doc_summary, _ = self.split_docstring(docstring)
            self.tasks.append(
                {
                    "task_type": "Function Explanation",
                    "input": f"Input: {code} \n What does this function do? \n",
                    "output": f"{summary}",
                    "code": code,
                    "function_name": row["func_name"],
                    "summary": summary,
                }
            )

            # Task 2: Parameter Explanation
            _, doc_details = self.split_docstring(docstring)
            self.tasks.append(
                {
                    "task_type": "Parameter Explanation",
                    "input": f"Input: {code} \n Explain the parameters of this function: \n",
                    "output": f"{doc_details}",
                    "code": code,
                    "function_name": row["func_name"],
                    "summary": summary,
                }
            )

            # Task 3: Summary
            self.tasks.append(
                {
                    "task_type": "Summary",
                    "input": f"Input: {code} \n Summarize this function in one sentence: \n",
                    "output": f"{summary}",
                    "code": code,
                    "function_name": row["func_name"],
                    "summary": summary,
                }
            )

            # Task 4: Docstring Completeness
            cut_point = random.randint(len(doc_summary), len(docstring))
            partial_docstring = docstring[:cut_point]
            is_complete = "Yes" if cut_point == len(docstring) else "No"
            missing_info = "" if is_complete == "Yes" else docstring[cut_point:].strip()

            self.tasks.append(
                {
                    "task_type": "Docstring Completeness",
                    "input": f"Input: {code}\n{partial_docstring} \n Is this docstring complete? If no, what's missing? \n",
                    "output": f"{is_complete}. {missing_info}",
                    "code": code,
                    "function_name": row["func_name"],
                    "summary": summary,
                }
            )

    def save_tasks(self) -> None:
        """
        Save the generated tasks as Parquet and CSV files.

        This method saves all tasks to a single Parquet file and a CSV file (limited to 100 rows)
        in the 'Data/' directory.
        """
        df_tasks = pd.DataFrame(self.tasks)

        os.makedirs("Data", exist_ok=True)

        # Save as Parquet
        table = pa.Table.from_pandas(df_tasks)
        pq.write_table(table, "Data/all_tasks.parquet")

        df_tasks.head(100).to_csv("Data/all_tasks_sample.csv", index=False)
