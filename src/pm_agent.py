import re
import os
import pandas

from src.loaders import file_loader, model_loader
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate


class PMAgent:

    def __init__(self, file_paths, prompt_template, config):
        self.vector_store = None
        self.retrieval_qa = None
        self.config = config
        self.prompt_template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.model = model_loader.get_model(self.config.reasoner_model_name, max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature, do_sample=self.config.do_sample, top_p=self.config.top_p)
        self.add_files(file_paths)

    def add_files(self, file_paths):
        file_paths = [self._preprocess_files(file_paths)]
        if self.vector_store is None:
            self.vector_store = file_loader.build_vector_store(file_paths=file_paths, model_name=self.config.splitter_model_name, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)

        else:
            vector_store = file_loader.build_vector_store(file_paths=file_paths, model_name=self.config.splitter_model_name, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
            self.vector_store.merge_from(vector_store)

        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.config.k_retriever}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def ask(self, query):
        result = self.retrieval_qa.invoke({"query": query})
        return [doc.page_content for doc in result["source_documents"]], result["result"]

    def _preprocess_files(self, file_paths):
        output_file = "../datasets/temp_combined_log.txt"
        columns_list = []
        valid_files = [f for f in file_paths if os.path.exists(f)]
        for file_name in valid_files:
            df_temp = pandas.read_csv(file_name, nrows=0)
            columns_list.append(set(df_temp.columns))

        common_columns = list(set.intersection(*columns_list))
        required_columns = {"case", "timestamp", "activity"}
        if not required_columns.issubset(set(common_columns)):
            raise Exception(f"Error: Missing one of the required columns {required_columns}.")

        extra_columns = [col for col in common_columns if col not in required_columns]
        with open(output_file, "w", encoding="utf-8") as file:
            for file_name in valid_files:
                file.write(f"\n=========================================\n")
                file.write(f"  STORIES FROM LOG FILE: {file_name}\n")
                file.write(f"=========================================\n\n")
                df = pandas.read_csv(file_name, usecols=common_columns)
                df["timestamp"] = pandas.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values(by=["case", "timestamp"])
                grouped = df.groupby("case")
                for case_id, group in grouped:
                    file.write(f"The case '{case_id}' started its journey,")
                    first = True
                    for _, row in group.iterrows():
                        time_str = row["timestamp"]
                        act_str = row["activity"]
                        sentence = f"{'' if first else 'then'} at {time_str}, the activity '{act_str}' was performed "
                        first = False
                        if extra_columns:
                            extra_info = ", ".join([f"{col}: {row[col]}" for col in extra_columns])
                            sentence += f" (Additional info -> {extra_info})"

                        file.write(sentence + ".")

                    file.write(f"The case '{case_id}' was then completed.\n\n")

        return output_file
