with open('template.html') as f:
    text = f.read()
with open('nodes.json') as f:
    data = f.read()

text = text.replace('{DATA}', data)
with open('nodes.html','w') as f:
    f.write(text)
