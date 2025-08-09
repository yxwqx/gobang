import requests

def ksm(x):
    result = 1
    while (x[1] > 0):
        if (x[1] & 1):
            result = result * x[0] % x[2]
        x[1] = int(x[1] / 2)
        x[0] = x[0] * x[0] % x[2]
    return result

answer = ''
getHtml = requests.get("http://183.175.12.27:8004/step_04/")
for i in eval(getHtml.json()['questions']):
    answer += str(ksm(i)) + ','
param = {'ans': answer[:-1]}
getHtml = requests.get("http://183.175.12.27:8004/step_04/", params=param)
print(getHtml.text)