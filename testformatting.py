import re

def parse_chat_completion(input_line):
    # Extract the content from the input line using regex
    # Adjusted regex to handle nested single quotes and newlines
    content_match = re.search(r"content='(.*?)(?<!\\)'", input_line, re.DOTALL)
    if content_match:
        content = content_match.group(1)
        print("Extracted content before processing:")  # Debugging
        print(content)  # Debugging
        
        content = content.replace('\\n', '\n')
        content = content.replace("\\'", "'")
        content = content.replace("\\\\", "\\")
        
        print("Extracted content after processing:")  # Debugging
        print(content)  # Debugging
        return content
    else:
        raise ValueError("Could not extract content from the input line.")

def save_as_markdown(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write(content)

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        input_line = file.readline().strip()
    
    print("Input line read from file:")  # Debugging
    print(input_line)  # Debugging
    
    content = parse_chat_completion(input_line)
    save_as_markdown(content, output_file)

if __name__ == "__main__":
    input_file = "testoutput.txt"
    output_file = "testoutput.md"
    main(input_file, output_file)