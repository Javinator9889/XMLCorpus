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
from collections import defaultdict
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pylatexenc.latexencode import UnicodeToLatexEncoder

# type hints
from typing import (
    Optional,
    Dict,
    List,
    Any,
    Union,
    TypeVar,
    Generic,
    Tuple,
    Set
)

encoder: UnicodeToLatexEncoder = \
    UnicodeToLatexEncoder(unknown_char_policy='replace',
                          replacement_latex_protection="braces",
                          non_ascii_only=True)


@dataclass
class XMLItem(ABC):
    """
    Base XML wrapper class. This item consists on a dataclass
    with basically two fields:
     + `tag`, containing the XML tag identifier.
     + `item_tag`, containing the XML tag itself.

    This abstract class defines two abstract methods that must be
    override:
     - :func:`parse`
     - :func:`to_table`

    Its main function is to simplify and contain basic XML data types.
    """

    tag: Optional[str]
    """
    The XML tag identifier, overriden by subclasses.
    """

    item_tag: str
    """
    The XML tag itself, overriden by subclasses.
    """

    @staticmethod
    @abstractmethod
    def parse(element: etree._Element,
              tag: str,
              **kwargs) -> "Optional[XMLItem]":
        """
        With the given :class:`lxml.etree.Element`, parses the :attr:`item_tag`
        and creates a new :class:`XMLItem` with its data.

        :param lxml.etree._Element element: the element to parse.
        :param str tag: the XML tag itself.
        :param kwargs: arbitrary arguments for custom parsing options.
        :return: the new tag.
        :rtype: XMLItem
        :raises ValueError: if the `element.tag` is different than `tag`.
        """

    @abstractmethod
    def to_table(self, tabletype="simple") -> str:
        """
        Represents the :class:`XMLItem` by a table.

        :param str tabletype: the table format to use. The following formats
            are available:
             + "plain"
             + "simple"
             + "github"
             + "grid"
             + "fancy_grid"
             + "pipe"
             + "orgtbl"
             + "jira"
             + "presto"
             + "pretty"
             + "psql"
             + "rst"
             + "mediawiki"
             + "moinmoin"
             + "youtrack"
             + "html"
             + "latex"
             + "latex_raw"
             + "latex_booktabs"
             + "textile"

        .. seealso::
           Table formats are defined by :mod:`tabulate` - more information
           about formatting at: https://pypi.org/project/tabulate/

        :return: the table representation of the :class:`XMLItem`.
        :rtype: str
        """

    @staticmethod
    def check_tag(element: etree._Element, tag: str):
        if element.tag != tag:
            raise ValueError(f"Element {element.tag}'s tag must be '{tag}'")

    def __eq__(self, other):
        """
        Checks if another :class:`XMLItem` equals to us.

        :param XMLItem other: the other item to check.

        :return: `True` if items share :attr:`tag` and :attr:`item_tag`,
            `False` otherwise.
        :rtype: bool
        """
        return (isinstance(other, self.__class__) and
                self.tag == other.tag and self.item_tag == other.item_tag)

    def __hash__(self):
        """
        Generates a unique representation of this object. Uses both :attr:`tag`
        and :attr:`item_tag` for this purpose.

        :return: the class hash.
        :rtype: int
        """
        return hash((self.tag, self.item_tag.hash))


T = TypeVar('T')
"""
Generic type for designating groups of XML tags.
"""


@dataclass
class XMLGroup(XMLItem, Generic[T]):
    """
    Specialization of :class:`XMLItem` for containing a variable set
    of fields of type :data:`T`.

    Those fields can be accessed in three ways:
     + By providing the index using :attr:`fields`.
     + By providing the field tag by using :attr:`dirs` and :attr:`fields`.
     + By direct access using both index or tag.
    """

    cls: T
    """
    The generic class used when parsing found subclasses.
    """

    subitem_tag: str = field(default=None, init=False)
    """
    The containing :data:`T` tag.
    """

    fields: List[T] = field(default_factory=list)
    """
    List of arbitrary length containing the :data:`T` objects.
    """

    dirs: Dict[str, int] = field(default_factory=dict)
    """
    Map containing the :data:`T` identifiers and its position in :attr:`fields`.
    """

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None,
              **kwargs) -> "Optional[XMLGroup]":
        """
        With the given :class:`lxml.etree.Element`, parses the :attr:`item_tag`
        and creates a new :class:`XMLGroup` with its data. In addition to
        :class:`XMLItem`, finds and parses any subitem contained by the tag.

        :param lxml.etree._Element element: the element to parse.
        :param T subcls: the subclass type used when parsing found objects.
        :param str tag: the XML tag itself.
        :param kwargs: arbitrary arguments for custom parsing options.
        :return: the new group of tags.
        :rtype: XMLItem
        :raises ValueError: if the `element.tag` is different than `tag`.
        :raises AttributeError: if `subcls` is not a subclass of
            :class:`XMLItem` or :class:`XMLGroup`.

        """
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

    def __eq__(self, other):
        return super(XMLGroup, self).__eq__(other) and \
               self.subitem_tag == other.subitem_tag

    @abstractmethod
    def to_table(self, tabletype="simple") -> str:
        pass

    def __getitem__(self, item: Union[str, int]):
        return self.fields[self.dirs[item]] if isinstance(item, str) \
            else self.fields[item]

    def __hash__(self):
        return hash((self.tag, self.item_tag, self.subitem_tag))


@dataclass
class Value(XMLItem):
    """
    The simplest XML item available, containing both a :attr:`tag` and a
    :attr:`summary`.
    """

    tag: str
    """
    :class:`Value` identifier tag.
    """

    summary: str
    """
    :class:`Value` summary.
    """

    item_tag: str = field(default="value", init=False, hash=hash('value'))

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

    def __eq__(self, other):
        return super(Value, self).__eq__(other) and \
               self.tag == other.tag and self.summary == other.summary


@dataclass
class Field(XMLGroup[Value]):
    """
    Class grouping a set of :class:`Value`s.
    """
    cls: T = Value
    item_tag: str = field(default='field', init=False, hash=hash('field'))
    subitem_tag: str = field(default='value', init=False, hash=hash('value'))

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
    """
    The morphology contains a group of fields containing values. This
    describes how the text's tokens are.
    """

    cls: T = Field
    item_tag: str = field(default='morphology', init=False, hash=hash('morph'))
    subitem_tag: str = field(default='field', init=False, hash=hash('field'))

    def to_table(self, ignored="simple") -> str:
        return str(self)

    def get(self, item: Union[str, int], default_value: Any = None) -> \
            Union[Field, Any]:
        """
        Searchs for an item, given its position or its tag. If not found,
        returns the default value.

        :param item: the item to look for. Can be the index
            or the identifier tag.
        :type item: str or int
        :param Any default_value: the value to return when not found.

        :return: the found :class:`Field` or the default value.
        :rtype: Field or Any
        """
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
    """
    Master class containing all possible annotations that can exist in a XML
    file.
    """

    morphology: Morphology
    """
    The annotation's morphology.
    """

    parts_of_speech: Optional[Field] = field(default=None)
    """
    The annotation's part of speech - can be `None`.
    
    :type: Field or None
    """

    gloss: Optional[Field] = field(default=None)
    """
    The annotation's glossary - can be `None`.
    
    :type: Field or None
    """

    def __str__(self):
        return self.to_table(tabletype="fancy_grid")

    @staticmethod
    def parse(annotation: etree._Element,
              tag: str = 'annotation',
              **kwargs) -> "Annotation":
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

    def __hash__(self):
        return hash((hash(self.morphology), self.parts_of_speech, self.gloss))


class AnnotationStatus(Enum):
    """
    Enumeration containing the three possible statuses for a sentence:
     1. Annotated
     2. Unannotated
     3. Reviewed
    """
    ANNOTATED = "annotated"
    UNANNOTATED = "unannotated"
    REVIEWED = "reviewed"


class AnnotationElements(Enum):
    """
    Enumeration containing the possible parts that conforms an annotation.
    Can be:
     1. Morphology
     2. Parts of speech
     3. Gloss
    """
    Morphology = "morphology"
    PartsOfSpeech = "part_of_speech"
    Gloss = "gloss"


def create_column_headers(first_header: str, tabletype: str) -> List[str]:
    """
    With the given first header and the table type, creates a list of headers
    used when designing the table for showing :class:`XMLItem` or
    :class:`XMLGroup` values.

    The output list consists on:
    .. code-block:: python
        return [[first header],
         [Lemma],
         [Part of speech],
         [Morphology],
         [Gloss]]

    :param str first_header: the first header to put.
    :param str tabletype: the table format - used only if LaTeX.

    :return: a list containing the headers.
    :rtype: list[str]
    """
    endcol = '|' if "plain" in tabletype else ''
    return [f"{first_header}\t\t{endcol}",
            f"Lemma\t\t{endcol}",
            f"Part of speech\t{endcol}",
            f"Morphology\t{endcol}",
            f"Gloss\t\t{endcol}"]


@dataclass
class Token(XMLItem):
    """
    The token represents a word. A word has only two mandatory attributes:
     + The `id`.
     + The `form`, it is, the word itself.

    All other values are optional and can be omitted.
    """
    id: str
    """
    The word unique ID.
    """

    form: str
    """
    The word itself.
    """

    alignment_id: Optional[List[str]] = None
    """
    Optional alignment ID, it is, the translated word(s) ID(s).
    """

    lemma: Optional[str] = None
    """
    Word's lemma.
    """

    part_of_speech: Optional[Value] = None
    """
    Optional part of speech corresponding that word.
    """

    morphology: Optional[Morphology] = None
    """
    Optional morphology items defining that word.
    """

    gloss: Optional[Value] = None
    """
    Optional glossary defined by that word.
    """

    tag: str = field(default=None, init=False)
    item_tag: str = field(default='token', init=False, hash=hash('token'))

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
                        except (KeyError, IndexError):
                            if i >= len(annotation.morphology.fields):
                                warn(f"More morphologies ({i + 1}) than "
                                     f"previously declared in annotations ("
                                     f"were "
                                     f"{len(annotation.morphology.fields)})")
                            else:
                                warn(f"Morphology with tag '{field}' not "
                                     f"found in field "
                                     f"'{annotation.morphology[i].tag}' ("
                                     f"token with ID: '{token.id}')")
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
        """
        Generates a list with the description of the word. It consists on:
         + Form.
         + Lemma.
         + Morphology fields.
         + Part of speech.
         + Glossary.

        :param str tabletype: the output format for the table - only used if
            LaTeX.
        :return: the token representation.
        :rtype: list[str]
        """
        form = f"\\textbf{{{self.form}}}" if "latex" in tabletype else self.form
        lemma = f"\\textit{{{self.lemma}}}" \
            if "latex" in tabletype else self.lemma or ''
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
        headers = create_column_headers(f"Word ({self.id})", tabletype)
        table_output = [[header] if add_headers else [] for header in headers]
        token_desc = self.describe(tabletype)
        for i, desc in zip(range(len(token_desc)), token_desc):
            table_output[i].append(desc)

        align = ("center",) * len(table_output[0])
        table = tabulate(table_output, colalign=align, tablefmt=tabletype)
        table = table.replace('{tabular}{¢', '{tabular}{c|')
        return encoder.unicode_to_latex(table) \
            if "latex" in tabletype \
            else table

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.tag == other.tag and self.item_tag == other.item_tag)

    def __hash__(self):
        return hash((self.tag, hash(self.item_tag)))


@dataclass
class Sentence(XMLGroup[Token]):
    """
    Structure containing a set of tokens, which conforms a sentence.
    """

    id: str = ""
    """
    Sentence unique ID.
    """

    cls = Token
    item_tag: str = field(default='sentence', init=False, hash=hash('sentence'))
    subitem_tag: str = field(default='token', init=False, hash=hash('token'))
    status: AnnotationStatus = field(default=AnnotationStatus.UNANNOTATED)
    """
    Sentence annotation status - possible values defined at 
    :class:`AnnotationStatus`.
    """

    alignment_id: Optional[str] = None
    """
    Aligned sentence ID - represents a translation of this sentence.
    """

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None,
              **kwargs) -> "Optional[Sentence]":
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
        if "latex" in tabletype:
            table = table.replace("{tabular}{c", "{tabular}{c|")
            return encoder.unicode_to_latex(table)
        else:
            return table

    def find_by(self, data: Dict[AnnotationElements, Union[Set[str], str]]) -> \
            List[Token]:
        """
        Recursively looks for tokens that fulfill with the data requirements
        specified.

        :param data: a dictionary containing the annotation elements to filter
            and the conditions of the filtering.
        :type data: dict[AnnotationElements, set[str] or str]
        :return: a list of tokens that fulfills the requirements.
        :rtype: list[Token]
        """
        found_tokens = defaultdict(set)
        keys = set()
        for token in self.fields:
            for element, topology in data.items():
                keys |= topology if isinstance(topology, set) else {topology}
                attr = getattr(token, element.value)
                if attr is not None:
                    if isinstance(attr, Morphology):
                        if isinstance(topology, str):
                            field, tag = topology.split('.', maxsplit=2)
                            if attr[field] and attr[field].tag == tag:
                                found_tokens[topology] |= {token}
                        else:
                            for topo in topology:
                                field, tag = topo.split('.', maxsplit=2)
                                if attr[field] and attr[field].tag == tag:
                                    found_tokens[topo] |= {token}
                    else:
                        if isinstance(topology, set):
                            raise ValueError(
                                "Data can be a set only when matching"
                                "morphology")
                        if attr.tag == topology:
                            found_tokens[topology] |= {token}
        tokens = set()
        for i, key in zip(range(len(keys)), keys):
            token = found_tokens[key]
            if len(tokens) == 0 and i == 0:
                tokens = token
            else:
                tokens &= token
        return list(tokens)

    def side_by_side(self,
                     another: "Sentence",
                     tabletype="plain") -> str:
        """
        With the given sentence, compares all tokens contained in both
        sentences (defined by their alignment ID) and generates a table
        with the comparison.

        :param Sentence another: the other sentence to compare.
        :param str tabletype: the output table format.

        :return: table representation of the comparison.
        :rtype: str
        """
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

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Source(XMLGroup[Sentence]):
    """
    The source conforms a set of sentences organized and translated into
    another source.
    """

    id: str = ''
    """
    Source unique ID.
    """

    language: str = ''
    """
    Source's language.
    """

    title: str = ''
    """
    Source's title.
    """

    citation_part: str = ''
    """
    Source's citation.
    """

    item_tag: str = field(default='source', init=False, hash=hash('source'))
    cls: T = Sentence
    subitem_tag: str = field(default='sentence',
                             init=False,
                             hash=hash('sentence'))
    alignment_id: Optional[str] = None
    """
    Source's translation's ID.
    """

    editorial_note: Optional[str] = None
    """
    Source's editorial note.
    """

    annotator: Optional[str] = None
    """
    Source's annotator.
    """

    reviewer: Optional[str] = None
    """
    Source's reviewer.
    """

    original_url: Optional[str] = None
    """
    Source's original URL.
    """

    @classmethod
    def parse(cls,
              element: etree._Element,
              subcls: T,
              tag: str = None,
              **kwargs) -> "Optional[Source]":
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
        header = encoder.unicode_to_latex(
            f"Source ``{self.id}''\n"
            f"\\begin{{itemize}}\n"
            f"\\item Language: {self.language}\n"
            f"\\item Aligned text ID: {self.alignment_id}\n"
            f"\\item Title: {self.title}\n"
            f"\\item Citation: {self.citation_part}\n"
            f"\\item Editorial note: {self.editorial_note}\n"
            f"\\item Annotator: {self.annotator}\n"
            f"\\item Reviewer: {self.reviewer}\n"
            f"\\item Original URL: \\url{{{self.original_url}}}\n"
            f"\\end{{itemize}}") \
            if "latex" in tabletype else \
            f"Source ``{self.id}''\n" \
            f"---------------------------------------\n" \
            f"\t Language: {self.language}\n" \
            f"\t Aligned text ID: {self.alignment_id}\n" \
            f"\t Title: {self.title}\n" \
            f"\t Citation: {self.citation_part}\n" \
            f"\t Editorial note: {self.editorial_note}\n" \
            f"\t Annotator: {self.annotator}\n" \
            f"\t Reviewer: {self.reviewer}\n" \
            f"\t Original URL: {self.original_url}\n" \
            f"#######################################"
        sentences = [header]
        for sentence in self.fields:
            sentences.append(sentence.to_table(tabletype))
        return '\n\n'.join(sentences)

    def compare(self, another: "Source",
                sentences: Tuple[str, ...] = (),
                status: Optional[AnnotationStatus] = None,
                tabletype: str = "simple") -> str:
        """
        With the given source, compares each sentence defined at `sentences`
        and generates a table with the sentences comparison.

        :param Source another: the other source to compare with.
        :param sentences: the sentences to compare. Empty means all.
        :type sentences: tuple[str, ...]
        :param AnnotationStatus status: the sentence status to use when
            comparing. None means unused.
        :param str tabletype: the output format for the table.

        :return: sources comparison as a table.
        :rtype: str

        :raises ValueError: if the sources are not aligned.
        """
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
        return '\n\n'.join(tables)

    def find_words_by(self,
                      data: Dict[AnnotationElements, Union[Set[str], str]]) -> \
            List[Token]:
        """
        With the given requirements, find all tokens that fulfills them.

        :param data: a dictionary containing the annotation elements to filter
            and the conditions of the filtering.
        :type data: dict[AnnotationElements, set[str] or str]
        :return: a list of tokens that fulfills the requirements.
        :rtype: list[Token]
        """
        results = []
        for field in self.fields:
            results.extend(field.find_by(data))
        return results


def main(args):
    """
    Main function that demonstrates how XMLCorpus works. Must receive a file
    containing two souces with IDs 'text1' and 'text2', respectively.

    :param args: command line arguments provided when this script is called.
    """
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
    print(text1.compare(text2, tabletype="grid"))
    for token in text1.find_words_by({
        AnnotationElements.Morphology: {"number.s", "gender.m"},
        AnnotationElements.PartsOfSpeech: 'Ne'
    }):
        print(token.to_table(tabletype="grid"))


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
