from dotenv import load_dotenv
from unstract.llmwhisperer import LLMWhispererClientV2
from pydantic import BaseModel, Field
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


class Item(BaseModel):
    name: str = Field(..., description="Name of the specific balance sheet item")
    amount: int = Field(None, description="The monetary value of the item")


class MajorCategory(BaseModel):
    name: str = Field(..., description="The name of the category (e.g., Current assets)")
    accounts: list[Item] = Field(description="The list of items within this category.")


class BalanceSheet(BaseModel):
    categories: list[MajorCategory] = Field(..., description="The list of major categories in the balance sheet")


def extract_text_from_pdf(file_path: str, pages: int) -> str:
    llmw = LLMWhispererClientV2(logging_level="INFO")
    result = llmw.whisper(file_path=file_path, pages_to_extract=str(pages+1), wait_for_completion=True)
    extracted_text = result["extraction"]['result_text']
    return extracted_text


def extract_text_from_pdf_with_llamaparse(file_path: str, pages: int, result_type: str = "markdown") -> str:
    parser = LlamaParse(
        result_type=result_type,
        target_pages=str(pages)
    )

    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    extracted_text = ''
    for doc in documents:
        extracted_text += doc.text

    return extracted_text


def complile_structuralizaiton_prompt(asset_breakdown: str):
    instruction = (
        "Your job is extracting hierarchical data of Asset breakdown from the given balance sheet, "
        "where the structure is determined by the indentation (tab spaces)."
        "Each level of indentation represents a deeper level in the hierarchy."
    )
    postamble = "Do not include any explanation in the reply. Only include the extracted information in the reply."
    system_tmplate = "{instruction}"
    human_template = "{format_instruction}\n\n{asset_breakdown}\n\n{postamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_tmplate)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    parser = PydanticOutputParser(pydantic_object=BalanceSheet)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    massages = chat_prompt.format_prompt(
        instruction=instruction, postamble=postamble, asset_breakdown=asset_breakdown, format_instruction=parser.get_format_instructions()
    ).to_messages()
    return massages


def compile_asset_extraction_prompt(extracted_text: str, target_year: int):
    preamble = (
        "You are seeing the page of annual report including a balance sheet, "
        "which is ordinarily divided into three main categories: assets and liabilities and equity.\n"
        "Your job is extracting the asset section from the given balance sheet, "
        f"but include only the data corresponding to the year {target_year}."
    )
    postamble = (
        "Do not make any changes to the format, indentation, structure, or content of the text. "
        "Simply copy the specified part without adding or removing any characters, whitespace, or punctuation.\n"
        "Ensure the output remains identical to the original."
    )
    system_template = "{preamble}"
    human_template = "{extracted_text}\n\n{postamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    messages = chat_prompt.format_prompt(preamble=preamble, postamble=postamble, extracted_text=extracted_text).to_messages()
    return messages


def extract_balance_sheet(file_path: str, pages: int, target_year: int, llamaparse: bool = True) -> BalanceSheet:
    if llamaparse:
        load_dotenv()
        extracted_text = extract_text_from_pdf_with_llamaparse(file_path, pages, result_type="text")
    else:
        extracted_text = extract_text_from_pdf(file_path, pages)

    client = ChatOpenAI(model='gpt-4o', temperature=0.0, max_retries=2)
    # Exctact Asset breakdown
    asset_extraction_prompt = compile_asset_extraction_prompt(extracted_text, target_year=target_year)
    asset_breakdown = client.invoke(asset_extraction_prompt).content
    print(asset_breakdown)

    # Structuralization
    structuralization_prompt = complile_structuralizaiton_prompt(asset_breakdown)
    structured_asset_breakdown = client.invoke(structuralization_prompt, response_format={"type": "json_object"}).content
    return structured_asset_breakdown


