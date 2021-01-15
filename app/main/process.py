import re, shlex, base64, io

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from wordcloud import WordCloud

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .reader import load_dir

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

def series2cloud(data):
    text = '\n\n'.join(data)
    cloud = WordCloud().generate(text)
    return cloud

def series2cloud_img(data):
    cloud = series2cloud(data)
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

def build_results(articles, chunks, matches, levnames, unit,
                  analysis_regexes, n_clusters):
    res = dict()

    nlev = len(levnames)
    if nlev == 0:
        # Just a bunch of plain files
        raise NotImplementedError()
    else:
        res['articles_summary'] = summarize_by_level(articles, levnames, "Article Counts")
        res['chunks_summary'] = summarize_by_level(chunks, levnames, f"Count of {unit}")
        res['matches_summary'] = summarize_by_level(matches, levnames,
                                                    f"Count of {unit} matching analysis terms")
        
    res['wordcloud_all_img'] = series2cloud_img(chunks['Text'])
    res['wordcloud_analysis_img'] = series2cloud_img(matches['Text'])

    res['analysis_table'] = make_table(matches, table_id='breakdown')

    sent = chunks[['Sentiment Score']].describe() #.drop(index='count')
    res['chunks_sentiment_summary'] = make_table(sent)

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
    res['matches_sentiment_breakdown'] = '\n<br>\n'.join(by_level_text)

    # cluster
    # @TODO: Also cluster by each analysis term?
    # @OTOD: Also cluster separately for each level?
    # @TODO: allow Tfidf customization: stop_words, ngrams, etc.
    # @TODO: use fancier tokenization for Tfidf (and other stuff?) (see my analysis notebook)
    vec = TfidfVectorizer(min_df=0.005, max_df=0.98,
                          max_features=1000, sublinear_tf=True,
                          ngram_range=(1,2), stop_words='english')
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

    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', z='_pca_z',
                       color=levnames[0], hover=levnames[1:])
    fig.update_layout(title_text='All filtered units')
    res['scatter_all_3d'] = render_plotly(fig)

    # @TODO: Allow customizing the clustering algorithm
    # @TODO: customize number of clusters
    k = min(n_clusters,X.shape[0])
    clst = KMeans(n_clusters=k, random_state=42)
    clst.fit(X)
    chunks['_cluster'] = clst.labels_
    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', color='_cluster',
                       hover=levnames)
    res['cluster_2d'] = render_plotly(fig)
    fig = make_scatter(chunks, x='_pca_x', y='_pca_y', z='_pca_z',
                       color='_cluster',
                       hover=levnames)
    res['cluster_3d'] = render_plotly(fig)

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
        info['keywords'] = words[keywords[i,:]]
        info['reps'] = chunks.iloc[reps[:,i], :]['Text']
        
        idx = chunks['_cluster'] == i
        info['cloud'] = series2cloud_img(chunks.loc[idx, 'Text'])
        res['cluster_info'].append(info)

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


def process(dname, path, info):
    #@TODO: Check for saved articles/chunks with same parameters
    
    level_filters = [info[key] for key in sorted(info)
                     if re.match('level_select-level[0-9]+_filter', key)]
    for i in range(len(level_filters)):
        for j in range(len(level_filters[i])):
            level_filters[i][j] = level_filters[i][j].lower()

    fpat = build_regex(info['fterms'], info['fregex'], info['fcase'])
    apat = build_regex(info['aterms'], info['aregex'], info['acase'])
    spat = re.compile(info['article_regex_splitter'], flags=re.M)

    articles_df, chunks_df = load_dir(path, level_filters, info['unit'], fpat, spat)

    colmap = {f'_level_{i}': name for i, name in enumerate(info['level_names'])}
    articles_df.rename(columns=colmap, inplace=True)
    chunks_df.rename(columns=colmap, inplace=True)

    chunks_df = chunks_df.reindex([*info['level_names'], 'Filename', 'Article ID', 'Text'], axis=1)
    print(articles_df.shape)
    print(chunks_df.shape)
    
    analyzer = SentimentIntensityAnalyzer()
    def score(row):
        return analyzer.polarity_scores(row['Text'])['compound']
    chunks_df['Sentiment Score'] = chunks_df.apply(score, axis=1)

    matches_df = chunks_df.query('Text.str.contains(@apat)')

    # @TODO: We're assuming that if the user chose regex for analysis
    # terms, they did not use a space...
    flags = re.I if info['aregex'] else 0
    analysis_regexes = {term: re.compile(f'\\b{term}\\b', flags=flags)
                        for term in info['aterms'].split()}

    #@TODO: Save articles/chunks after filtering, plus parameters used

    #@TODO: Save html results
    
    return build_results(articles_df, chunks_df, matches_df,
                         info['level_names'], info['unit'], analysis_regexes,
                         info['n_clusters'])
    
