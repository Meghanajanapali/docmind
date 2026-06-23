from embedding import model

text = "React hooks manage state"

embedding = model.encode(text)

print(embedding)