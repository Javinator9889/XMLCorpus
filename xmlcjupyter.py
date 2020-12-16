#                             XMLCorpus
#                  Copyright (C) 2020 - Javinator9889
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#                   (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#               GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
from lxml import etree
from io import BytesIO
from ipywidgets import widgets
from IPython.core.display import HTML
from IPython.display import display, Latex
from typing import Dict, Tuple, Union, Optional
from xmlc import (
    Source,
    Annotation,
    Sentence,
    AnnotationStatus,
    AnnotationElements
)


def init():
    display(HTML("<style>pre { white-space: pre !important; }</style>"))


def show_widgets():
    uploader = widgets.FileUpload(accept='xml', multiple=False)
    url_box = widgets.Text(placeholder='Put URL to XML here...',
                           description='URL: ',
                           disabled=False)

    display(uploader, url_box)
    return uploader, url_box


def annotations_widget():
    annotations_box = widgets.Checkbox(value=False,
                                       description='Use additional '
                                                   'annotations file',
                                       disabled=False)
    annotations_uploader = widgets.FileUpload(accept='xml',
                                              multiple=False)
    annotations_url = widgets.Text(placeholder='Put URL to XML here...',
                                   description='Annotation XML file: ',
                                   disabled=False)

    display(annotations_box, annotations_uploader, annotations_url)
    return annotations_box, (annotations_uploader, annotations_url)


def load_source(file: widgets.FileUpload = None,
                url: widgets.Text = None,
                use_annotations_file: widgets.Checkbox = None,
                annotations_data: Tuple[
                    widgets.FileUpload, widgets.Text] = None):
    if file is None and url is None:
        raise ValueError("Either file or URL must be given")

    data = url.value \
        if url is not None and url.value != '' \
        else BytesIO(file.value[file.metadata[0]['name']]['content'])
    parser = etree.XMLParser(remove_comments=True)

    if use_annotations_file is not None:
        if use_annotations_file.value:
            if annotations_data is None:
                raise ValueError(
                    '"Use additional annotations file" was checked '
                    'but no file provided')
            annotations_file, annotations_url = annotations_data
            if annotations_file is None and annotations_url is None:
                raise ValueError(
                    '"Use additional annotations file" was checked '
                    'but no file provided')
            annotation_data = annotations_url.value \
                if annotations_url is not None and annotations_url.value != '' \
                else BytesIO(annotations_file
                             .value[annotations_file.metadata[0]['name']][
                                 'content'])
        else:
            annotation_data = None
    else:
        annotation_data = None

    tree = etree.parse(data, parser=parser)
    annotations_tree = etree.parse(annotation_data, parser=parser) \
        .find('annotation') \
        if annotation_data is not None \
        else tree.find('annotation')
    return annotations_tree, tree


def load_annotations(tree: etree._Element) -> Annotation:
    return Annotation.parse(tree)


def parse_tree(tree: etree._Element,
               annotation: Annotation) -> Dict[str, Source]:
    sources = {}
    for source in tree.findall('source'):
        src = Source.parse(source,
                           subcls=Sentence,
                           annotation=annotation,
                           ignored_tags={'title'})
        sources[src.id] = src

    return sources


def show_table(table, tabletype):
    display(Latex(table)) if "latex" in tabletype else display(table)


def display_source(source: Union[Dict[str, Source], Source],
                   tabletype="latex_raw"):
    def show_source(src: Source):
        print(src.to_table(tabletype))

    if isinstance(source, dict):
        for src in source.values():
            show_source(src)
    else:
        show_source(source)


def compare(source: Source,
            another_source: Source,
            sentences: Tuple[str, ...] = (),
            status: Optional[AnnotationStatus] = None,
            tabletype="latex_raw"):
    comparison = source.compare(another_source,
                                sentences=sentences,
                                status=status,
                                tabletype=tabletype)
    print(comparison)


def find_words_by(data: Dict[AnnotationElements, Union[set, str]],
                  source: Union[Dict[str, Source], Source],
                  tabletype="latex_raw"):
    if isinstance(source, Source):
        source = {'': source}
    for src in source.values():
        tokens = src.find_words_by(data)
        for token in tokens:
            print(token.to_table(tabletype))
