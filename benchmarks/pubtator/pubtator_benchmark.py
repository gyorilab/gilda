import time
import requests

url_prefix = 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/'
submit_url = url_prefix + 'submit/'
retrieve_url = url_prefix + 'retrieve/'


def process_file(filename):
    begin = time.time()
    data = open(filename, 'rb').read()
    res = requests.post(submit_url + 'Gene', data=data)
    session_key = res.text
    delay = 10
    res = requests.get(retrieve_url + session_key)
    while res.status_code != 200:
        res = requests.get(retrieve_url + session_key)
        print('Waiting %.2fs' % delay)
        time.sleep(delay)
    end = time.time()
    print('Overall time: %s' % (end - begin))
    return end - begin


if __name__ == '__main__':
    nrep = 10
    times = []
    for i in range(nrep):
        print('Starting iteration %d' % i)
        tt = process_file('pubtator_input.json')
        times.append(tt)
    print('Average time: %s' % (sum(times) / nrep))