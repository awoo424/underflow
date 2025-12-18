*Stream of consciousness* is a narrative style that usually mimics the continuous flow of thoughts in a character's mind.

How does a machine learning model navigate between thoughts? 
This CLI application attempts to visualise the stream of consciousness inside a language model. Every input sentence is converted into a continuous stream of words by traversing from word to word in the vector space of the language model.

## How to use

- Enter text in the input area
- Alternatively, press `Generate example` to get example inputs. 
- The input must be **lowercase alphabetical characters and spaces**. 
- Press `Submit` to generate the stream-of-consciousness text using the input. 
- Press `Cancel` to stop the current generation process.
- Press `Clear output` to clear the output text.
- Press `Ctrl+S` to save the output text as a .txt file.
- Press `Ctrl+Q` or `Esc` to quit.

## Example

Input:
> hello world nice to meet you

The program will find a path from *hello* to *world*, from *world* to *nice*, and so on in the word vector space.

Output:
> **hello** hey you like **world** come good **nice** something get **to** will next **meet** next come **you**
