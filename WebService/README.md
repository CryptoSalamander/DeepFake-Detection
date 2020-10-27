## Instructions

```
pip3 install -r requirements.txt
```

## Run

```
./run
```

## Directory Structure :

	├── api
	│   ├── classifiers.py
	│   ├── DeepFake.py
	│   └── deepfake_utils.py
	├── app
	│   ├── forms.py
	│   ├── __init__.py
	│   ├── models.py
	│   ├── routes.py
	│   ├── static
	│   │   ├── avatar
	│   │   │   └── avatar.png
	│   │   ├── cover_pics
	│   │   │   └── cover.png
	│   │   ├── css
	│   │   │   └── main.css
	│   │   └── videos
	│   └── templates
	│       ├── account.html
	│       ├── home.html
	│       ├── layout.html
	│       ├── login.html
	│       ├── register.html
	│       ├── update_video.html
	│       ├── upload.html
	│       └── video.html
	├── config.py
	├── migrations
	│   ├── alembic.ini
	│   ├── env.py
	│   ├── README
	│   ├── script.py.mako
	│   └── versions
	│       └── b0e1bb55d78d_create_models.py
	├── model
	│   └── TrustNet_fp16.trt
	├── README.md
	├── requirements.txt
	├── run
	├── server.py
	├── youtube.code-workspace
	└── youtube.db

