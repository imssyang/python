import srt

data = """
1
00:00:08,780 --> 00:00:09,820
怎么回事
What just happened?

2
00:00:09,930 --> 00:00:12,730
我刚才还在听那个奇怪的磁带  然后就到这里了
I was just listening to that strange cassette, and now I'm here.

3
00:00:13,560 --> 00:00:14,650
我晕过去了吗
Did I pass out?

4
00:00:15,180 --> 00:00:16,450
这是我们的老木屋
This is our old cabin.
"""

for sub in srt.parse(data):
    print(sub)
