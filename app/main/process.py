import os, random, time
import re, shlex, base64, io, json, uuid, nltk
from celery.app.registry import TaskRegistry
from requests import post
from zipfile import ZipFile
from pathlib import Path
from chardet import UniversalDetector
from multiprocessing import Process

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from scipy.sparse import data
from wordcloud import WordCloud

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from app import celery, socketio
from .reader import load_raw_articles, load_wrangled
from .util import is_supported_filename

detector = UniversalDetector()
def guess_encoding(of):
  detector.reset()
  for line in of:
    detector.feed(line)
    if detector.done: break
  detector.close()
  if detector.result['confidence'] < 0.5:
    # print('\n--- Low confidence!---')
    # print(detector.result)
    return None
  # if ascii, just use utf-8 in case it's wrong
  encoding = detector.result['encoding']
  if encoding == 'ascii':
    return 'utf-8'
  # sometimes chardet makes weird guesses for Windows, like Windows-1254 (Turkish)...
  if encoding.startswith('Windows-'):
    return 'Windows-1252'
  return encoding

CHUNK_SIZE = 1024*1024
def iter_binary_buffered(binary, encoding):
  # @TODO: Instead of silently replacing encoding errors, it would be
  # nice to at least print/log a warning somewhere. But we also don't
  # want to have to re-encode everything if the error happened near
  # the end of the file. To do this, I think you want to look at:
  # codecs.register_error()
  with io.TextIOWrapper(binary, encoding, errors='replace') as wrapper:
    while True:
      chunk = wrapper.read(CHUNK_SIZE)
      if not chunk: return
      yield chunk

def binary_to_file(binary, encoding, path):
  with path.open('w') as output:
    for chunk in iter_binary_buffered(binary, encoding):
      output.write(chunk)

def make_dataset(home, zfile, wait_time=2):
    # (home / '.error').unlink(missing_ok=True) # only 3.8
    errfile = home / '.error'
    if errfile.exists():
        errfile.unlink()
    #p = Process(target=extract, args=(home, zfile))
    #p.start()
    task = extract.delay(os.fspath(home), os.fspath(zfile))
    # allow some time to finish, in which case we can take the user
    # directly to the next stage.
    #p.join(timeout=wait_time)
    return task

@celery.task()
def extract(home, zfile):
    try:
        os.mkdir(home)
        zf = ZipFile(zfile, 'r')

        for filename in zf.namelist():
            path = home / Path(filename)
            if filename.endswith('/'):
                # we want to explicitly make these, since we want to keep empty directories.
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not is_supported_filename(path):
                print("Skipping unsupported file: ", filename)
                continue

            # Might not have gotten to the directory yet in the for loop...
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Start ", filename, end='')

            if filename.endswith('.txt'):
                with zf.open(filename) as binary:
                    encoding = guess_encoding(binary)
                print(f' ({encoding}) ', end='')
                with zf.open(filename) as binary:
                    binary_to_file(binary, encoding, path)
            else:
                zf.extract(filename, home)

            print('\t...done!')
    except Exception as e:
        print("Preprocessing failed!")
        print(e)
        error = '.error'
        with open(os.path.join(home, error), 'w') as f:
            f.write(str(e))
        raise

    print(f"Successfully preprocessed {zfile} into {home}")
    file = '.preprocessed'
    # just touch a file to indicate we are done
    with open(os.path.join(home, file), 'w') as f:
        return True


def summarize_by_level(df, levnames, title=None, histfunc='count', z=None):
    nlev = len(levnames)
    assert nlev > 0
    graph = px.density_heatmap(data_frame=df, x=levnames[0],
                               y=levnames[1] if 1 < nlev else None,
                               facet_row=levnames[2] if 2 < nlev else None,
                               facet_col=levnames[3] if 3 < nlev else None,
                               title=title, histfunc=histfunc, z=z)
    return render_plotly(graph)

def render_plotly(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)

def series2cloud(data, stop_words):
    text = '\n\n'.join(data)
    cloud = WordCloud(stopwords = stop_words).generate(text)
    return cloud

def series2cloud_img(data, stop_words):
    cloud = series2cloud(data, stop_words)
    with io.BytesIO() as buffer:
        cloud.to_image().save(buffer, 'png')
        img = base64.b64encode(buffer.getvalue()).decode()
    return img

def make_scatter(df, x, y, color=None, hover=None, z=None):
    fig = go.Figure()
    marker = dict()

    if z:
        marker['size'] = 6
        scatfunc = go.Scatter3d
    else:
        marker['opacity'] = 0.5
        scatfunc = go.Scattergl

    if color:
        groups = [(group, df[df[color] == group]) for group in df[color].unique()]
        hcols = [color]
    else:
        groups = [(None, df)]
        hcols = []

    if hover:
        hcols += hover
    else:
        htext = None

    for group, data in groups:
        kwarg = dict()
        if z: kwarg['z'] = data[z]
        if group is not None: kwarg['name'] = str(group)
        if hover:
            labels = [c.replace('_','').capitalize()+': '+data[c].astype(str).values for c in hcols]
            htext = ['<br>'.join(row) for row in zip(*labels)]

        fig.add_trace(scatfunc(x=data[x], y=data[y], mode='markers', **kwarg,
                               marker=marker, hovertext=htext, hoverinfo='text'))


    # @TODO: Set default zoom level or resize; 3D plots start out too
    # small/too far away
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig

def make_table(df, table_id=None, classes=None):
    if classes is None:
        classes = ['table']
    return df.to_html(table_id=table_id, classes=classes).replace(r'\n', '<br>')


def build_default_stopwords():
    sciswords = set(text.ENGLISH_STOP_WORDS)
    nltkswords = set(nltk.corpus.stopwords.words('english'))
    default_stopwords = sciswords.union(nltkswords)
    return sorted(default_stopwords)

def build_stopwords(words, default_words, use_default):
    extra_words = words.split()
    extra_words = set(extra_words)
    if use_default:
        stopwords = set(default_words)
        stopwords = stopwords.union(extra_words)
    else:
        stopwords = extra_words
    print(stopwords)
    return(stopwords)

def build_results(articles, chunks, matches, levnames, unit,
                  analysis_regexes, n_clusters, swords, defaultswords, uid):
    res = dict()

    defswords= build_default_stopwords()
    stopwords = build_stopwords(swords, defswords, defaultswords)

    nlev = len(levnames)
    if nlev == 0:
        # Just a bunch of plain files
        raise NotImplementedError()
    else:
        res['articles_summary'] = summarize_by_level(articles, levnames, "Article Counts")
        socketio.emit('taskprogress', res, to=uid)
        res['chunks_summary'] = summarize_by_level(chunks, levnames, f"Count of {unit}")
        socketio.emit('taskprogress', res, to=uid)
        res['matches_summary'] = summarize_by_level(matches, levnames,
                                                    f"Count of {unit} matching analysis terms")
        socketio.emit('taskprogress', res, to=uid)

    res['wordcloud_all_img'] = series2cloud_img(chunks['Text'], stopwords)
    socketio.emit('taskprogress', res, to=uid)
    res['wordcloud_analysis_img'] = series2cloud_img(matches['Text'], stopwords)
    socketio.emit('taskprogress', res, to=uid)

    res['analysis_table'] = make_table(matches, table_id='breakdown')
    socketio.emit('taskprogress', res, to=uid)

    sent = chunks[['Sentiment Score']].describe() #.drop(index='count')
    res['chunks_sentiment_summary'] = make_table(sent)
    socketio.emit('taskprogress', res, to=uid)

    sent = matches[['Sentiment Score']].describe()
    sent.rename(columns={'Sentiment Score': 'All terms'}, inplace=True)
    by_level_text = ['<br><br>\n']
    for term, regex in analysis_regexes.items():
        this = matches.query('Text.str.contains(@regex)')
        sent[term] = this['Sentiment Score'].describe()

        bylev = f"{term}: No matches!" if len(this) == 0 else \
                summarize_by_level(this, levnames, title=term,
                                   histfunc='avg', z='Sentiment Score')
        by_level_text.append(bylev)

    res['matches_sentiment_summary'] = make_table(sent)
    socketio.emit('taskprogress', res, to=uid)
    res['matches_sentiment_breakdown'] = '\n<br>\n'.join(by_level_text)
    socketio.emit('taskprogress', res, to=uid)

    # cluster
    # @TODO: Also cluster by each analysis term?
    # @OTOD: Also cluster separately for each level?
    # @TODO: allow Tfidf customization: stop_words, ngrams, etc.
    # @TODO: use fancier tokenization for Tfidf (and other stuff?) (see my analysis notebook)
    vec = TfidfVectorizer(min_df=0.005, max_df=0.98,
                          max_features=1000, sublinear_tf=True,
                          ngram_range=(1,2), stop_words=stopwords)
    X = vec.fit_transform(chunks['Text'])
    # @TODO: Allow changing PCA settings
    dimred = PCA(n_components=3, svd_solver='arpack')
    Xproj = dimred.fit_transform(X.toarray())
    chunks['_pca_x'] = Xproj[:,0]
    chunks['_pca_y'] = Xproj[:,1]
    chunks['_pca_z'] = Xproj[:,2]

    fig = make_scatter(chunks, x='_pca_x', y='_pca_y',
                       color=levnames[0], hover=levnames[1:])
    fig.update_layout(title_text='All filtered units')
    res['scatter_all_2d'] = render_plotly(fig)
    socketio.emit('taskprogress', res, to=uid)

    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', z='_pca_z',
                       color=levnames[0], hover=levnames[1:])
    fig.update_layout(title_text='All filtered units')
    res['scatter_all_3d'] = render_plotly(fig)
    socketio.emit('taskprogress', res, to=uid)

    # @TODO: Allow customizing the clustering algorithm
    # @TODO: customize number of clusters
    k = min(n_clusters,X.shape[0])
    clst = KMeans(n_clusters=k, random_state=42)
    clst.fit(X)
    chunks['_cluster'] = clst.labels_
    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', color='_cluster',
                       hover=levnames)
    res['cluster_2d'] = render_plotly(fig)
    socketio.emit('taskprogress', res, to=uid)
    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', z='_pca_z',
                       color='_cluster',
                       hover=levnames)
    res['cluster_3d'] = render_plotly(fig)
    socketio.emit('taskprogress', res, to=uid)

    # words important to each cluster
    keywords = np.argsort(clst.cluster_centers_, axis=1)[:,-10:]

    # closest chunks to each cluster center
    dist = clst.transform(X)
    reps = np.argsort(dist, axis=0)[:5,:]

    res['cluster_info'] = []
    words = np.array(vec.get_feature_names())
    for i in range(k):
        info = dict()
        idx = keywords[i,:]
        info['keywords'] = list(words[keywords[i,:]])
        info['reps'] = list(chunks.iloc[reps[:,i], :]['Text'])

        idx = chunks['_cluster'] == i
        info['cloud'] = series2cloud_img(chunks.loc[idx, 'Text'], stopwords)
        res['cluster_info'].append(info)
    socketio.emit('taskprogress', res, to=uid)
    return res

# We're just going to treat everything as a regex, so escape it if necessary.
def build_regex(s, use_regex, case, flags=0):
    if not s: return None
    if use_regex:
        # @TODO: allow multiple regex terms
        pat = s
    else:
        # Hopefully these do not need to be human-readable, b/c
        # re.escape escapes many unnecessary characters...
        terms = [re.escape(x) for x in shlex.split(s)]
        base = '|'.join(terms)
        pat = f'\\b(?:{base})\\b'

    rflags = flags if flags else 0
    if case: rflags |= re.I
    return re.compile(pat, flags=rflags)

def process(dname, path, form, uid):
    #@TODO: Check for saved articles/chunks with same parameters

    #level_filters = [form[key] for key in sorted(form)
                     #if re.match('level_select-level[0-9]+_filter', key)]
    #for i in range(len(level_filters)):
        #for j in range(len(level_filters[i])):
            #level_filters[i][j] = level_filters[i][j].lower()

    #fpat = build_regex(form['fterms'], form['fregex'], form['fcase'])
    #apat = build_regex(form['aterms'], form['aregex'], form['acase'])

    # @TODO: We're assuming that if the user chose regex for analysis
    # terms, they did not use a space...
    #flags = re.I if form['aregex'] else 0
    #analysis_regexes = {term: re.compile(f'\\b{term}\\b', flags=flags)
                        #for term in form['aterms'].split()}

    outdir = path / '.output'
    outdir.mkdir(exist_ok=True)


    outfile = outdir / uid
    if outfile.exists():
        outfile.unlink()

    # (home / '.error').unlink(missing_ok=True) # only 3.8
    errfile = path / '.error'
    if errfile.exists():
        errfile.unlink()

    #args = uid, path, form['level_names'], level_filters, form['unit'], \
           #fpat, apat, analysis_regexes, form['n_clusters']
    #p = Process(target=explore, args=args)
    socketio.emit('switchtab', to=uid)
    task = explore.delay(uid, os.fspath(path), form['level_names'], form, form['unit'], form['fterms'], form['fregex'], form['fcase'], \
        form['aterms'], form['aregex'], form['acase'], form['n_clusters'], form['swords'], form['defaultswords'])
    #p.start()
    # allow some time to finish, in which case we can take the user
    # directly to the next stage.
    #p.join(timeout=3)
    return uid

@celery.task()
def explore(uid, path, level_names, form, uoa,
            fterms, fregex, fcase, aterms, aregex, acase, n_clusters, swords, defaultswords):
    print(f'Begin explore "{path}": {uid}')
    level_filters = [form[key] for key in sorted(form)
                     if re.match('level_select-level[0-9]+_filter', key)]
    for i in range(len(level_filters)):
        for j in range(len(level_filters[i])):
            level_filters[i][j] = level_filters[i][j].lower()
    fpat = build_regex(fterms, fregex, fcase)
    apat = build_regex(aterms, aregex, acase)
    flags = re.I if form['aregex'] else 0
    analysis_regexes = {term: re.compile(f'\\b{term}\\b', flags=flags)
                        for term in form['aterms'].split()}

    articles_df, chunks_df = load_wrangled(path, level_filters, uoa, fpat)

    print("Finished loading data:")
    print('\t Articles: ', articles_df.shape)
    print('\t Chunks: ', chunks_df.shape)

    # @TODO: write to log file
    if chunks_df.shape[0] == 0:
        res = dict()
        res['error'] = "Nothing matched the filter terms!"
        print(res['error'])
        output = '.output'
        outfile = os.path.join(path, output, uid)
        with open(outfile, 'w') as f:
            json.dump(res, f)
        return

    colmap = {f'_level_{i}': name for i, name in enumerate(level_names)}
    articles_df.rename(columns=colmap, inplace=True)
    chunks_df.rename(columns=colmap, inplace=True)

    chunks_df = chunks_df.reindex([*level_names, 'Filename', 'Article ID', 'Text'], axis=1)

    analyzer = SentimentIntensityAnalyzer()
    def score(row):
        return analyzer.polarity_scores(row['Text'])['compound']
    chunks_df['Sentiment Score'] = chunks_df.apply(score, axis=1)

    matches_df = chunks_df.query('Text.str.contains(@apat)')

    #@TODO: Save articles/chunks after filtering, plus parameters used

    res = build_results(articles_df, chunks_df, matches_df, level_names,
                        uoa, analysis_regexes, n_clusters, swords, defaultswords, uid)
    output = '.output'
    outfile = os.path.join(path, output, uid)
    with open(outfile, 'w') as f:
        json.dump(res, f)

    print(f'"{path} results built and dumped: {uid}')
    socketio.emit('taskstatus', res, to=uid)
    return True

def wrangle_dataset(path, oneper, splitter, use_regex, level_names, level_vals):
    vals = [list(x) for x in level_vals]
    names = [name if name else f'level{i}' for i,name in enumerate(level_names)]
    if oneper:
        pat = None
    elif use_regex:
        pat = splitter
    else: # plain text splitter, treat as regex anyway
        pat = re.escape(splitter)

    # (path / '.error').unlink(missing_ok=True) # only 3.8
    errfile = path / '.error'
    if errfile.exists():
        errfile.unlink()

    #p = Process(target=parse_articles, args=(path, names, vals, pat))
    #p.start()
    task = parse_articles.delay(os.fspath(path), names, vals, pat)
    # allow some time to finish, in which case we can take the user
    # directly to the next stage.
    #p.join(timeout=2)
    return True

@celery.task()
def parse_articles(path, level_names, level_vals, splitter):
    print(f"Wrangling started: {path}")
    try:
        regex = re.compile(splitter, flags=re.M)
        articles = load_raw_articles(path, level_names, regex)
        wrangled = '.wrangled.pkl'
        articles.to_pickle(os.path.join(path, wrangled))
    except Exception as e:
        print("Wrangling failed!")
        print(e)
        error = '.error'
        with open(os.path.join(path, error), 'w') as f:
            f.write(str(e))
        raise

    print(f"Successfully wrangled {path}")
    print(articles.shape)
    # save some info for later (no longer used since we pickled)
    # also indicates to the rest of the server that we are done wrangling
    info = dict(level_names=level_names, level_vals=level_vals,
                article_regex_splitter=splitter)
    jsonpath = '.info.json'
    with open(os.path.join(path, jsonpath), 'w') as f:
        json.dump(info, f)
