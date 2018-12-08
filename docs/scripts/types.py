import json


class Media(object):
    __slots__ = ('publisher', 'title', 'month', 'day', 'year', 'video', 'url')

    def __init__(self, g):
        self.publisher = g[0]
        self.title = g[1]
        self.month = g[2]
        self.day = g[3]
        self.year = g[4]
        self.video = True if g[5].find('video') >= 0 else False
        self.url = g[6]

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def smart_title(s):
    return ' '.join(w if w.isupper() else w.capitalize() for w in s.split())


def build_media(g):
    return {
        'publisher': g[0],
        'title': smart_title(g[1]),
        'month': g[2],
        'day': g[3],
        'year': g[4],
        'video': True if g[5].find('video') >= 0 else False,
        'url': g[6],
    }


def build_student(g):
    return {
        'url': g[0],
        'photo': g[1],
        'name': g[2],
        'title': 'Ph.D. Student',
        'category': 'Graduate Student',
        'time': '',
        'department': 'Computer Science',
        'current': 'University of Maryland',
        'year': 2013,
    }


def build_papers(g):
    return {
        'bibname': 'tmp',
        'url': g[0].strip().strip('.').replace('/papers/', ''),
        'title': g[1].strip().strip('.'),
        'bib': g[2].strip().strip('.').replace('/papers/BibTex/', ''),
        'author': g[3].strip().strip('.'),
        'type': 'article',
        'booktitle': g[4].strip().strip('.'),
        'attr': g[5].strip().strip('.'),
        'pages': '',
        'year': '2018',
        'month': '',
        'day': '',
        'series': '',
        'book_title_short': '',
        'editor': '',
        'isbn': '',
        'numpages': 1,
        'doi': '',
        'publisher': '',
        'address': '',
        'location': '',
        'volume': 1,
        'number': 1,
        'keywords': '',
        'note': '',
        'citations': '',
        'published': True,
        'visible': True,
        'poseter': '',
        'video': '',
        'youtube': '',
        'vimeo': '',
        'code': '',
        'slides': '',
        'talk': '',
        'award': '',
        'abstract': '',
        'image': 'empty.jpg',
        'supple': '',
        'categories': '',
    }
