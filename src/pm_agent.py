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
        return result['result']
