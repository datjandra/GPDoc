import os
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha", token=os.environ['HF_TOKEN'])
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")

docs = SimpleDirectoryReader("patient_data").load_data()
index = VectorStoreIndex.from_documents(docs, service_context=service_context)
query_engine = index.as_query_engine()

gpdoc = "GPDoc:"
exit_conditions = (":q", "quit", "exit", "bye")
print("Enter one of these keywords to exit: ", exit_conditions)
while True:
  query = input("> ")
  if query in exit_conditions:
    print(gpdoc, "Good bye")
    exit(1)
  else:
    answer = query_engine.query(query)
    if answer:
      print(gpdoc, answer.response.strip())
    else:
      print(gpdoc, "Sorry I can't help you with that. Try rephrasing your question.")