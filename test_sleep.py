import time
start = time.time()
frames = 0
while frames < 300:
    frames += 1
    expected = frames / 30.0
    curr = time.time() - start
    if curr < expected:
        time.sleep(expected - curr)
print("Effective FPS:", frames / (time.time() - start))
