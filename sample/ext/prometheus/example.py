import http.server
import random
import time
from prometheus_client import start_http_server
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Summary
from prometheus_client import Histogram

REQUESTS = Counter(
    "server_requests_total", "Total number of requests to this webserver"
)
EXCEPTIONS = Counter(
    "server_exceptions_total", "Total number of exception raised by this webserver"
)
PROGRESS = Gauge("server_requests_inprogress", "Number of requests in progress")
REQUEST = Gauge("server_last_request_time", "Last request start time")
RESPONSE = Gauge("server_last_response_time", "Last request serve time")
LATENCY = Summary("server_latency_seconds", "Time to serve a web page")
LATENCY_BLOCK = Summary("server_latency_block_seconds", "Time to run a block of code")
LATENCY2 = Histogram(
    "server_latency2_seconds",
    "Time to serve a web page",
    buckets=[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
)


class ServerHandler(http.server.BaseHTTPRequestHandler):
    @LATENCY2.time()
    @EXCEPTIONS.count_exceptions()
    @PROGRESS.track_inprogress()
    def do_GET(self):
        time_request = time.time()
        REQUEST.set_to_current_time()
        REQUESTS.inc()

        """
        if random.random() > 0.5:
            raise Exception

        with EXCEPTIONS.count_exceptions(ValueError):
            if random.random() > 0.5:
                raise ValueError
        """

        """
        PROGRESS.inc()
        rand_value = random.random()
        if rand_value > 0.7:
            PROGRESS.dec()

        if rand_value > 0.1 and rand_value < 0.2:
            PROGRESS.set(0)
            print("PROGRESS reset")
        """

        with PROGRESS.track_inprogress():
            print("Starting to sleep...")
            time.sleep(0.1)
            print("It's time to work")

        time.sleep(random.random() / 10)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello World!")

        RESPONSE.set_to_current_time()
        time_response = time.time()
        LATENCY.observe(time_response - time_request)

    @LATENCY.time()
    def do_POST(self):
        with LATENCY_BLOCK.time():
            print("Starting to sleep...")
            time.sleep(0.1)
            print("Waking up")

        time.sleep(random.random())
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello World!")


if __name__ == "__main__":
    start_http_server(8000)
    server = http.server.HTTPServer(("", 8001), ServerHandler)
    print("Prometheus metrics available on port 8000 /metrics")
    print("HTTP server available on port 8001")
    server.serve_forever()
