# 请求库
import requests
from bs4 import BeautifulSoup

url = 'https://qn.bjx.com.cn/'
# url = 'https://www.taobao.com'
# url = 'https://news.cnblogs.com/'

# 用于模拟浏览器访问的header，不用这个会被forbidden
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'}
# 爬取的网页链接
r = requests.get(url, headers=headers)
# 类型
# print(type(r))
print(r.status_code)
# 中文显示
# r.encoding='utf-8'
r.encoding = None
# print(r.encoding)
# print(r.text)
result = r.text

bs = BeautifulSoup(result, 'html.parser')
# 具体标签
# 获取已爬取内容中的td标签内容
div_list = bs.find_all('div')
# print(ul_list)
dicts = {}

for div_node in div_list:
    # class="wdwlist"
    dicts = div_node.attrs
    if dicts is not None and len(dicts) > 0 and dicts.get('class') is not None and dicts.get('class')[0] == "wdwlistnr":
        print('查询到节点wdwlist{}'.format(div_node))
