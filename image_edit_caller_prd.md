Here's a PRD for the program that tests the API capabilities of a model that does image editing. The model accepts a single image or a number of images and a prompt, and produces a result in a form of another image. The model is hosted on OpenRouter and its name is "google/gemini-2.5-flash-image-preview". I also supply the API key that is currently temporarily hosted in the file `.env` here in this folder. The prompt is in the file `prompt.md`.

Project description

1. Testing phase
We need to test out the image generation from a single character from characters folder, and a single selfie from a selfies_samples folder
with the supplied prompt and supplied .env with openrouter API key. The API should return a picture, that has to be encoded and put into  
`results` folder.

2. After the aforementioned script, a page has to be generated. It should have a picture generated for each of the characters from `characters`
folder and for each selfie from `selfies_samples` folder - thus a 2d grid of pictures. The pictures are in `results/` folder and the table is in
md format.

Technical Details:

API Implementation:
- Endpoint: https://openrouter.ai/api/v1/chat/completions
- Authentication: Bearer token in Authorization header using API key from .env
- Image format: Base64 encoded images in the content array
- Request structure: Messages array with content parts (text and image_url)
- Model: "google/gemini-2.5-flash-image-preview"

Error Handling:
- Retry failed requests up to 3 times
- Display actual error messages in table cells for failed generations
- Log necessary debugging information

Output Requirements:
- Save generated images with naming convention: {original_filename}_result.png
- Track and display response time for each image generation
- Include timing metrics in the markdown table cells along with results
- Results stored in `results/` folder
- Generate markdown table showing 2D grid of all results
