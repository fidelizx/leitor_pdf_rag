from groq import Groq

client = Groq(api_key="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")

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Olá! Teste de API Groq."}]
)

print(resp.choices[0].message.content)
