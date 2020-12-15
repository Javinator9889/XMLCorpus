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
from warnings import warn
from tabulate import tabulate
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pylatexenc.latexencode import UnicodeToLatexEncoder

# type hints
from typing import Optional, Dict, List, Any, Union, TypeVar, Generic, Tuple

encoder: UnicodeToLatexEncoder = \
    UnicodeToLatexEncoder(unknown_char_policy='replace',
                          replacement_latex_protection="braces",
                          non_ascii_only=True)


@dataclass
class XMLItem(ABC):
    tag: Optional[str]
    item_tag: field(init=False, default=None)

    @staticmethod
    @abstractmethod
    def parse(element: etree._Element,
              tag: str,
              **kwargs) -> "Optional[XMLItem]":
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
              tag: str = None,
              **kwargs) -> "Optional[XMLGroup]":
        if not issubclass(subcls, XMLItem) and not issubclass(subcls, XMLGroup):
            raise AttributeError(f"Class {subcls} must inherit from XMLItem or "
                                 f"XMLGroup")
        fields = list()
        dirs = dict()
        idx = 0
        for field in element:
            if issubclass(subcls, XMLGroup):
                value = subcls.parse(element=field,
                                     subcls=subcls.cls,
                                     tag=tag or subcls.subitem_tag,
                                     **kwargs)
            else:
                value = subcls.parse(element=field,
                                     tag=tag or subcls.item_tag,
                                     **kwargs)
            if value is not None:
                fields.insert(idx, value)
                dirs[value.tag] = idx
                idx += 1

        return cls(element.get('tag'), element.tag, fields, dirs)

    @abstractmethod
    def to_table(self, tabletype="simple") -> str:
        pass

    def __getitem__(self, item: Union[str, int]):
        return self.fields[self.dirs[item]] if isinstance(item, str) \
            else self.fields[item]


@dataclass
class Value(XMLItem):
    tag: str
    summary: str
    item_tag: str = field(default="value", init=False)

    @staticmethod
    def parse(element: etree._Element, tag: str = 'value', **kwargs) -> "Value":
        XMLItem.check_tag(element, tag)
        return Value(element.get('tag'), element.get('summary'))

    def to_table(self, tabletype="simple") -> str:
        table = tabulate([(self.tag, self.summary)],
                         headers=('Tag', 'Summary'),
                         tablefmt=tabletype)
        return encoder.unicode_to_latex(table) \
            if "latex" in tabletype \
            else table


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
        tab = tabulate(table,
                       headers=('Field tag', 'Values'),
                       tablefmt=tabletype,
                       colalign=("center",))
        return encoder.unicode_to_latex(tab) \
            if "latex" in tabletype \
            else tab

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
              tag: str = 'annotation', **kwargs) -> "Annotation":
        XMLItem.check_tag(annotation, tag)
        morphology = Morphology.parse(element=annotation.find('morphology'),
                                      subcls=Morphology.cls)

        pos = annotation.find('parts-of-speech')
        if pos is not None:
            parts_of_speech = Field.parse(pos, subcls=Value, **kwargs)

        gls = annotation.find('gloss')
        if gls is not None:
            gloss = Field.parse(gls, subcls=Value, **kwargs)

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
        table = '\n'.join(res)
        return encoder.unicode_to_latex(table) \
            if "latex" in tabletype \
            else table


class AnnotationStatus(Enum):
    ANNOTATED = "annotated"
    UNANNOTATED = "unannotated"
    REVIEWED = "reviewed"


def create_column_headers(first_header: str, tabletype: str) -> List[str]:
    endcol = '' if "latex" in tabletype else '|'
    return [f"{first_header}\t\t{endcol}",
            f"Lemma\t\t{endcol}",
            f"Part of speech\t{endcol}",
            f"Morphology\t{endcol}",
            f"Gloss\t\t{endcol}"]


@dataclass
class Token(XMLItem):
    id: str
    form: str
    alignment_id: Optional[List[str]] = None
    lemma: Optional[str] = None
    part_of_speech: Optional[Value] = None
    morphology: Optional[Morphology] = None
    gloss: Optional[Value] = None
    tag: str = field(default=None, init=False)
    item_tag: str = field(default='token', init=False)

    @staticmethod
    def parse(element: etree._Element,
              tag: str = 'token',
              **kwargs) -> "XMLItem":
        XMLItem.check_tag(element, tag)
        token = Token(id=element.get('id'), form=element.get('form'))
        annotation: Annotation = kwargs['annotation']
        for attr, value in element.attrib.items():
            attr_value = value
            if attr == 'morphology':
                fields = []
                dirs = {}
                idx = 0
                for i, field in zip(range(len(value)), value):
                    if field != '-':
                        try:
                            fields.insert(idx, annotation.morphology[i][field])
                            dirs[annotation.morphology[i].tag] = idx
                            idx += 1
                        except KeyError:
                            if i > len(annotation.morphology.fields):
                                warn(f"More morphologies {i} than previously"
                                     f"declared (were "
                                     f"{len(annotation.morphology.fields)})")
                            else:
                                warn(f"Non-identified morphology item "
                                     f"'{field}' at position "
                                     f"'{annotation.morphology[i].tag}'")
                attr_value = Morphology(tag=None, fields=fields, dirs=dirs)
            elif attr == 'part-of-speech':
                attr_value = annotation.parts_of_speech[value]
            elif attr == 'gloss':
                attr_value = annotation.gloss[value]
            elif attr == 'alignment-id':
                attr_value = value.split(',')
            attr = attr.replace('-', '_')
            setattr(token, attr, attr_value)
        token.tag = token.id
        return token

    def describe(self, tabletype="simple") -> List[str]:
        form = f"\\textbf{{{self.form}}}" if "latex" in tabletype else self.form
        lemma = f"\\textit{{{self.lemma}}}" \
            if "latex" in tabletype and self.lemma is not None else ''
        token_desc = [form, lemma]
        if self.part_of_speech is not None:
            token_desc.insert(2, self.part_of_speech.summary)
        else:
            token_desc.insert(2, '')
        if self.morphology is not None:
            desc_morph = []
            for value in self.morphology.fields:
                desc_morph.append(value.summary)
            token_desc.insert(3, ' '.join(desc_morph))
        else:
            token_desc.insert(3, '')
        if self.gloss is not None:
            token_desc.insert(4, self.gloss.summary)
        else:
            token_desc.insert(4, '')
        if "latex" in tabletype:
            for i in range(1, len(token_desc)):
                token_desc[i] = f"{{\\small {token_desc[i]}}}"
        return token_desc

    def to_table(self, tabletype="simple", add_headers=True) -> str:
        headers = create_column_headers(f"Word", tabletype)
        table_output = [[]] * len(headers)
        if add_headers:
            for i, header in zip(range(len(headers)), headers):
                table_output.insert(i, [header])
        token_desc = self.describe()
        for i, desc in zip(range(len(token_desc)), token_desc):
            table_output[i].append(desc)

        align = ("center",) * len(table_output[0])
        table = tabulate(table_output, colalign=align, tablefmt=tabletype)
        return encoder.unicode_to_latex(table) \
            if "latex" in tabletype \
            else table


@dataclass
class Sentence(XMLGroup[Token]):
    id: str = ""
    cls = Token
    item_tag: str = field(default='sentence', init=False)
    subitem_tag: str = field(default='token', init=False)
    status: AnnotationStatus = field(default=AnnotationStatus.UNANNOTATED)
    alignment_id: Optional[str] = None

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None,
              **kwargs) -> "Optional[XMLGroup]":
        if element.tag in kwargs['ignored_tags']:
            return None
        sentence = super(Sentence, cls).parse(element,
                                              subcls,
                                              tag,
                                              **kwargs)
        sentence.id = element.get('id')
        sentence.status = AnnotationStatus[element.get('status').upper()]
        sentence.alignment_id = element.get('alignment-id')
        sentence.tag = sentence.id

        return sentence

    def to_table(self, tabletype="plain") -> str:
        sentence_id_fmt = f"\\texttt{{{self.id}}}" if "latex" in tabletype \
            else self.id
        table_output = [[header] for header in create_column_headers(
            f"{sentence_id_fmt} ({self.status.value})", tabletype
        )]
        for token in self.fields:
            if token is None:
                continue
            desc = token.describe()
            for i, data in zip(range(len(desc)), desc):
                table_output[i].append(data)
        align = ("center",) * len(table_output[0])
        table = tabulate(table_output, colalign=align, tablefmt=tabletype)
        return encoder.unicode_to_latex(table) \
            if "latex" in tabletype \
            else table

    def side_by_side(self,
                     another: "Sentence",
                     tabletype="plain") -> str:
        if self.alignment_id != another.id and another.alignment_id != self.id:
            raise ValueError("Sentences are not aligned!")
        if self.alignment_id == another.id:
            source = self
            other = another
        else:
            source = another
            other = self

        sentence1_id_fmt = f"\\texttt{{{source.id}}}" if "latex" in tabletype \
            else source.id
        sentence2_id_fmt = f"\\texttt{{{other.id}}}" if "latex" in tabletype \
            else other.id
        headers = create_column_headers(
            f"{sentence1_id_fmt} ({source.status.value})", tabletype
        )
        headers.extend(create_column_headers(f"{sentence2_id_fmt} ("
                                             f"{other.status.value})",
                                             tabletype))
        table_output = [[header] for header in headers]
        for token in source.fields:
            if token is None:
                continue
            desc1 = token.describe(tabletype)
            for i, data in zip(range(len(desc1)), desc1):
                table_output[i].append(data)

            aligned_tokens = None
            aligned_token_ids = token.alignment_id or []
            for token_id in aligned_token_ids:
                other_token = other[token_id]
                desc2 = other_token.describe(tabletype)
                if aligned_tokens is None:
                    aligned_tokens = desc2
                else:
                    for i, data in zip(range(len(desc2)), desc2):
                        aligned_tokens[i] = f"{aligned_tokens[i]} - {data}"

            if aligned_tokens is None:
                aligned_tokens = [''] * len(desc1)

            start = len(desc1)
            stop = start + len(aligned_tokens)
            for i, data in zip(range(start, stop), aligned_tokens):
                table_output[i].append(data)
        align = ("center",) * len(table_output[0])
        table = tabulate(table_output, colalign=align, tablefmt=tabletype)
        if "latex" in tabletype:
            table = table.replace("{tabular}{c", "{tabular}{c|", 1) \
                .replace(f"\\\\\n \\texttt{{{other.id}}}",
                         f"\\\\[1ex]\n\\hline\n \\texttt{{{other.id}}}")
            return encoder.unicode_to_latex(table)
        return table


@dataclass
class Source(XMLGroup[Sentence]):
    id: str = ''
    language: str = ''
    title: str = ''
    citation_part: str = ''
    item_tag: str = field(default='source', init=False)
    cls: T = Sentence
    subitem_tag: str = field(default='sentence', init=False)
    alignment_id: Optional[str] = None
    editorial_note: Optional[str] = None
    annotator: Optional[str] = None
    reviewer: Optional[str] = None
    original_url: Optional[str] = None

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None,
              **kwargs) -> "XMLGroup":
        sid = element.get('id')
        language = element.get('language')
        alignment_id = element.get('alignment-id')
        title = element.find('title').text
        citation_part = element.find('citation-part').text
        editorial_note = element.find('editorial-note').text
        annotator = element.find('annotator').text
        reviewer = element.find('reviewer').text
        original_url = element.find('electronic-text-original-url').text
        source = super(Source, cls).parse(element.find('div'),
                                          subcls,
                                          tag,
                                          **kwargs)
        source.id = sid
        source.language = language
        source.alignment_id = alignment_id
        source.title = title
        source.citation_part = citation_part
        source.editorial_note = editorial_note
        source.annotator = annotator
        source.reviewer = reviewer
        source.original_url = original_url
        source.tag = sid

        return source

    def to_table(self, tabletype="simple") -> str:
        sentences = []
        for sentence in self.fields:
            sentences.append(sentence.to_table(tabletype))
        return '\n\n'.join(sentences)

    def compare(self, another: "Source",
                sentences: Tuple[str, ...] = (),
                status: Optional[AnnotationStatus] = None,
                tabletype: str = "simple") -> str:
        if self.alignment_id != another.id and another.alignment_id != self.id:
            raise ValueError("Sources are not aligned!")
        if self.alignment_id == another.id:
            source = self
            other = another
        else:
            source = another
            other = self
        tables = []
        for sentence1 in source.fields:
            aligned_sentence_id = sentence1.alignment_id
            sentence2 = other.fields[other.dirs[aligned_sentence_id]]
            if len(sentences) > 0:
                if sentence1.id not in sentences:
                    continue
            if status is not None:
                if sentence1.status != status or sentence2.status != status:
                    continue
            tables.append(sentence1.side_by_side(sentence2, tabletype))
        # print(self.dirs)
        # print(self.fields)
        # for sentence1 in source.fields:
        #     res = []
        #     # print(f"{key} - {idx}")
        #     # sentence1 = source.fields[idx]
        #     other_key = sentence1.alignment_id
        #     sentence2 = other.fields[another.dirs[other_key]]
        #     if len(sentences) > 0:
        #         if sentence1.id not in sentences:
        #             continue
        #     if status is not None:
        #         if sentence1.status != status or sentence2.status != status:
        #             continue
        #     sentence_tt = tabletype if "latex" in tabletype else "plain"
        #     res.append([sentence1.to_table(tabletype=sentence_tt)])
        #     res.append([sentence2.to_table(tabletype=sentence_tt)])
        #
        #     tables.append(tabulate(res, tablefmt=tabletype))
        return '\n\n'.join(tables)


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

    sources: Dict[str, Source] = {}
    for source in tree.findall('source'):
        src = Source.parse(source, Sentence,
                           annotation=annotation,
                           ignored_tags={'title'})
        sources[src.id] = src

    text1 = sources['text1']
    text2 = sources['text2']
    print(text1.compare(text2, tabletype="latex_raw"))
    # for _, source in sources.items():
    #     print(source.to_table(tabletype="latex_booktabs"))


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
