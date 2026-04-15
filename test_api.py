import requests
print("Health Check:")
print(requests.get('http://127.0.0.1:8000/health').json())

def test_api(idea, sector, country):
    payload = {'idea': idea, 'sector': sector, 'country': country}
    res = requests.post('http://127.0.0.1:8000/analyze', json=payload)
    data = res.json()
    if 'success' not in data:
        print(f'Error: {data}')
        return
    print(f"Sector: {sector}, Country: {country}")
    print(f"Idea length: {len(idea.split())} words")
    print(f"TAS Score: {data.get('tas_score')}, Idea Score: {data.get('idea_score')}")
    print(f"SVS: {data.get('svs')} -> Quadrant: {data.get('quadrant')}")
    print('-' * 40)

print('\nTesting 1: High Market, High Idea')
test_api('This solves a huge problem with an innovative SaaS subscription business model that can scale globally and provide great revenue.', 'Fintech', 'AE — UAE')

print('\nTesting 2: High Market, Low Idea')
test_api('hello world', 'Fintech', 'AE — UAE')

print('\nTesting 3: Low Market, High Idea')
test_api('This solves a huge problem with an innovative SaaS subscription business model that can scale globally and provide great revenue.', 'Edtech', 'EG — Egypt')

print('\nTesting 4: Low Market, Low Idea')
test_api('hello world', 'Edtech', 'EG — Egypt')
