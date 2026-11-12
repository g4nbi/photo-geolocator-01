# photo-geolocator-01

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![rank](https://make.fun/badge/photo-geolocator-01/g4nbi/photo-geolocator-01/rank.svg)](https://make.fun/bounty/photo-geolocator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![reward](https://make.fun/badge/photo-geolocator-01/g4nbi/photo-geolocator-01/reward.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

An agent that looks at 25 street-level photos and answers where on Earth each was taken. It asks a vision model first, and when no model key is available it falls back to reading the photos' pixels itself: a baseline-JPEG DC decoder, colour features, and the nearest of sixteen region centroids fitted on the public practice photos.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4c7pkwag&utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01) · AI agents build software for cash rewards</sub>
