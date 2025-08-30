Here's a PRD for the program that tests the API capabilities of a model that does image editing. The model accepts a single image or a number of images and a prompt, and produces a result in a form of another image. The model is hosted on OpenRouter and its name is "google/gemini-2.5-flash-image-preview". I also supply the API key that is currently temporarily hosted in the file `.env` here in this folder. The prompt is in the file `prompt.md`.

Project description

1. Testing phase
We need to test out the image generation from a single character from characters folder, and a single selfie from a selfies_samples folder
with the supplied prompt and supplied .env with openrouter API key. The API should return a picture, that has to be encoded and put into  
`results` folder.

2. After the aforementioned script, a page has to be generated. It should have a picture generated for each of the characters from `characters`
folder and for each selfie from `selfies_samples` folder - thus a 2d grid of pictures. The pictures are in `results/` folder and the table is in
md format.
