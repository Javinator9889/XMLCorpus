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
from enum import Enum
from lxml import etree
from tabulate import tabulate
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# type hints
from typing import Optional, Dict, List, Any, Union, TypeVar, Generic


@dataclass
class XMLItem(ABC):
    tag: str
    item_tag: field(init=False, default=None)

    @staticmethod
    @abstractmethod
    def parse(element: etree._Element, tag: str) -> "XMLItem":
        """Parses the element into the selected tag. This method must be
        implemented by all subclasses"""

    @abstractmethod
    def to_table(self, tabletype="simple") -> str:
        pass

    @staticmethod
    def check_tag(element: etree._Element, tag: str):
        if element.tag != tag:
            raise ValueError(f"Element {element.tag}'s tag must be '{tag}'")


T = TypeVar('T')


@dataclass
class XMLGroup(XMLItem, Generic[T]):
    cls: T
    subitem_tag: str = field(default=None, init=False)
    fields: List[T] = field(default_factory=list)
    dirs: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None) -> "XMLGroup":
        if not issubclass(subcls, XMLItem) and not issubclass(subcls, XMLGroup):
            raise AttributeError(f"Class {subcls} must inherit from XMLItem or "
                                 f"XMLGroup")
        fields = list()
        dirs = dict()
        idx = 0
        for field in element:
            if issubclass(subcls, XMLGroup):
                value = subcls.parse(field, subcls.cls,
                                     tag or subcls.subitem_tag)
            else:
                value = subcls.parse(field, tag or subcls.item_tag)
            fields.insert(idx, value)
            dirs[value.tag] = idx
            idx += 1

        return cls(element.get('tag'), element.tag, fields, dirs)

    @abstractmethod
    def to_table(self, tabletype="simple") -> str:
        pass


@dataclass
class Value(XMLItem):
    tag: str
    summary: str
    item_tag: str = field(default="value", init=False)

    @staticmethod
    def parse(element: etree._Element, tag: str = 'value') -> "Value":
        XMLItem.check_tag(element, tag)
        return Value(element.get('tag'), element.get('summary'))

    def to_table(self, tabletype="simple") -> str:
        return tabulate([(self.tag, self.summary)],
                        headers=('Tag', 'Summary'),
                        tablefmt=tabletype)


@dataclass
class Field(XMLGroup[Value]):
    cls: T = Value
    item_tag: str = field(default='field', init=False)
    subitem_tag: str = field(default='value', init=False)

    def to_table(self, tabletype="simple") -> str:
        table_contents = []
        for value in self.fields:
            table_contents.append((value.tag, value.summary))
        tmp_table = tabulate(table_contents,
                             headers=('Tag', 'Summary'),
                             tablefmt="plain")
        table = [(self.tag, tmp_table)]
        return tabulate(table,
                        headers=('Field tag', 'Values'),
                        tablefmt=tabletype,
                        colalign=("center",))

    def __getitem__(self, item: Union[str, int]):
        return self.fields[self.dirs[item]] if isinstance(item, str) \
            else self.fields[item]

    def __str__(self):
        return self.to_table("fancy_grid")


@dataclass
class Morphology(XMLGroup[Field]):
    cls: T = Field
    item_tag: str = field(default='morphology', init=False)
    subitem_tag: str = field(default='field', init=False)

    def to_table(self, ignored="simple") -> str:
        return str(self)

    def get(self, item: Union[str, int], default_value: Any = None) -> \
            Union[Field, Any]:
        try:
            return self.fields[item] if isinstance(item, int) \
                else self.fields[self.dirs[item]]
        except KeyError:
            return default_value

    def __getitem__(self, item: Union[str, int]):
        return self.get(item)

    def __str__(self):
        res = ['Morphology']
        for value in self.fields:
            res.append(str(value))
        return '\n'.join(res)


@dataclass
class Annotation(XMLItem):
    morphology: Morphology
    parts_of_speech: Optional[Field] = field(default=None)
    gloss: Optional[Field] = field(default=None)

    def __str__(self):
        return self.to_table(tabletype="fancy_grid")

    @staticmethod
    def parse(annotation: etree._Element,
              tag: str = 'annotation') -> "Annotation":
        XMLItem.check_tag(annotation, tag)
        morphology = Morphology.parse(element=annotation.find('morphology'),
                                      subcls=Morphology.cls)

        pos = annotation.find('parts-of-speech')
        if pos is not None:
            parts_of_speech = Field.parse(pos, subcls=Value)

        gls = annotation.find('gloss')
        if gls is not None:
            gloss = Field.parse(gls, subcls=Value)

        return Annotation(annotation.get('tag'),
                          annotation.tag,
                          morphology,
                          parts_of_speech,
                          gloss)

    def to_table(self, tabletype="simple") -> str:
        res = [str(self.morphology)]
        if self.parts_of_speech is not None:
            res.append('Parts of speech')
            pos = []
            for value in self.parts_of_speech.fields:
                pos.append((value.tag, value.summary))
            res.append(tabulate(pos,
                                headers=('Tag', 'Summary'),
                                tablefmt=tabletype,
                                colalign=("center", "center")))
        if self.gloss is not None:
            res.append('Gloss')
            gls = []
            for value in self.gloss.fields:
                gls.append((value.tag, value.summary))
            res.append(tabulate(gls,
                                headers=('Tag', 'Summary'),
                                tablefmt=tabletype,
                                colalign=("center", "center")))
        return '\n'.join(res)


class AnnotationStatus(Enum):
    ANNOTATED = "annotated"
    UNANNOTATED = "unannotated"
    REVIEWED = "reviewed"


@dataclass
class Token:
    id: str
    form: str
    alignment_id: Optional[List[str]] = None
    lemma: Optional[str] = None
    part_of_speech: Optional[Value] = None
    morphology: Optional[List[Morphology]] = None
    gloss: Optional[Value] = None


@dataclass
class Sentence:
    id: str
    status: AnnotationStatus = field(default=AnnotationStatus.UNANNOTATED)
    alignment_id: Optional[str] = None
    tokens: Dict[str, Token] = field(default_factory=dict)


@dataclass
class Source:
    id: str
    language: str
    title: str
    citation_part: str
    alignment_id: Optional[str] = None
    editorial_note: Optional[str] = None
    annotator: Optional[str] = None
    reviewer: Optional[str] = None
    original_url: Optional[str] = None
    sentences: Dict[str, Sentence] = field(default_factory=dict)

    @staticmethod
    def parse(source: etree.Element, annotations: Annotation) -> "Source":
        sid = source.get('id')
        language = source.get('language')
        alignment_id = source.get('alignment-id')
        title = source.find('title').text
        citation_part = source.find('citation-part').text
        editorial_note = source.find('editorial-note').text
        annotator = source.find('annotator').text
        reviewer = source.find('reviewer').text
        original_url = source.find('electronic-text-original-url').text

        sentences = dict()
        for sentence in source.find('div').findall('sentence'):
            sentence_id = sentence.get('id')
            sentence_aid = sentence.get('alignment-id')
            sentence_status = AnnotationStatus[sentence.get('status').upper()]
            tokens = dict()
            for token in sentence.findall('token'):
                tkid = token.get('id')
                form = token.get('form')
                tk_alignment_id = token.get('alignment-id').split(',') \
                    if token.get('alignment-id') is not None else None

                lemma = token.get('lemma')
                part_of_speech = annotations \
                    .parts_of_speech[token.get('part-of-speech')] \
                    if token.get('part-of-speech') is not None else None

                morphology = None
                mph = token.get('morphology')
                if mph is not None:
                    morphology = dict()
                    for i, morph in zip(range(len(mph)), mph):
                        current_morph = annotations.morphology[i]
                        found_morph = current_morph.values.get(morph, None)
                        if found_morph is not None:
                            morphology[found_morph.tag] = Morphology(
                                OrderedDict({
                                    current_morph.tag: Field(
                                        current_morph.tag, {
                                            found_morph.tag: found_morph
                                        })
                                })
                            )

                gloss = annotations.gloss[token.get('gloss')] \
                    if token.get('gloss') is not None else None

                tokens[tkid] = Token(tkid, form, tk_alignment_id, lemma,
                                     part_of_speech, morphology, gloss)

            sentences[sentence_id] = Sentence(sentence_id, sentence_status,
                                              sentence_aid, tokens)
        return Source(id=sid,
                      language=language,
                      alignment_id=alignment_id,
                      title=title,
                      citation_part=citation_part,
                      editorial_note=editorial_note,
                      annotator=annotator,
                      reviewer=reviewer,
                      original_url=original_url,
                      sentences=sentences)


def main(args):
    parser = etree.XMLParser(remove_comments=True)
    tree = etree.parse(args.file, parser=parser)
    annotation_element = None
    if args.annotation_file is not None:
        annotation_tree = etree.parse(args.annotation_file)
        annotation_element = annotation_tree.find('annotation')
    if annotation_element is None:
        annotation_element = tree.find('annotation')

    annotation = Annotation.parse(annotation_element)
    print(annotation)

    sources = dict()
    for source in tree.findall('source'):
        found_source = Source.parse(source, annotation)
        sources[found_source.id] = found_source

    for key, value in sources.items():
        table_output = []
        for _, sentence in value.sentences.items():
            sentence_line = [f"{sentence.id} ({sentence.status.value})\t|"]
            lemma_line = ["Lemma\t\t|"]
            pos_line = ["Part of speech\t|"]
            morph_line = ["Morphology\t|"]
            gloss_line = ["Gloss\t\t|"]
            for _, token in sentence.tokens.items():
                sentence_line.append(token.form)
                lemma_line.append(token.lemma or '')
                # lemma = f"{token.lemma}[/red]" \
                #     if token.lemma is not None else ''
                #
                # lemma_line.append(lemma)
                if token.part_of_speech is not None:
                    pos_line.append(token.part_of_speech.summary)
                else:
                    pos_line.append('')
                if token.morphology is not None and len(token.morphology) > 0:
                    desc_morph = []
                    # print(token.morphology)
                    for morph in token.morphology:
                        # print(morph)
                        # print(token.morphology[morph].fields)
                        for field in token.morphology[morph].fields:
                            # print(list(token.morphology[morph][
                            #              field].values.items())[0][1].summary)
                            desc_morph.append(
                                list(token.morphology[morph][
                                         field].values.items())[0][1].summary)
                        # desc_morph.append(token.morphology[morph].fields)
                    morph_line.append(' '.join(desc_morph))
                else:
                    morph_line.append('')
                if token.gloss is not None:
                    gloss_line.append(token.gloss.summary)
                else:
                    gloss_line.append('')

            table_output.append(sentence_line)
            table_output.append(lemma_line)
            table_output.append(pos_line)
            table_output.append(morph_line)
            table_output.append(gloss_line)
            break
        align = ("center",) * len(table_output[0])
        print(tabulate(table_output, colalign=align, tablefmt="plain"),
              end='\n\n')

        # console = Console()
        # console.print(tabulate(table_output, colalign=align,
        # tablefmt="plain"))

        # complete_sentence = []
        # for _, sentence in value.sentences.items():
        #     for _, token in sentence.tokens.items():
        #         complete_sentence.append(token.form)
        # print(' '.join(complete_sentence))

    # print(sources['text1'].sentences['0a'].tokens['2a'].morphology)


if __name__ == '__main__':
    parser = ArgumentParser(description="XMLCorpus file parser")
    parser.add_argument("file",
                        metavar="FILENAME",
                        help="XML file to analyze")
    parser.add_argument("-af",
                        "--annotation-file",
                        metavar="FILENAME",
                        help="Optional XML file containing annotation data",
                        default=None)
    main(parser.parse_args())
