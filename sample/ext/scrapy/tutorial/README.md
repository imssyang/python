# Init
$ scrapy startproject tutorial

# Run
$ cd ./tutorial
$ scrapy crawl quotes

# Shell
$ scrapy shell "https://quotes.toscrape.com/page/1/"

In [5]: response.css("title")
Out[5]: [<Selector query='descendant-or-self::title' data='<title>Quotes to Scrape</title>'>]

In [6]: response.css("title::text").getall()
Out[6]: ['Quotes to Scrape']

In [7]: response.css("title").getall()
Out[7]: ['<title>Quotes to Scrape</title>']

In [8]: response.css("title::text").get()
Out[8]: 'Quotes to Scrape'

In [9]: response.css("title::text")[0].get()
Out[9]: 'Quotes to Scrape'

In [10]: response.css("noelement").get()

In [11]: response.css("title::text").re(r"(\w+) to (\w+)")
Out[11]: ['Quotes', 'Scrape']

In [12]: response.xpath("//title")
Out[12]: [<Selector query='//title' data='<title>Quotes to Scrape</title>'>]

In [13]: response.xpath("//title/text()").get()
Out[13]: 'Quotes to Scrape'

# Store
scrapy crawl quotes -O quotes.json   # overwrite
scrapy crawl quotes -o quotes.jsonl  # append

