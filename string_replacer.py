import re


def identity(s):
    return s


def lowercase(s):
    return s.lower()


class StringReplacer(object):

    def __call__(self, string):
        """
        Process the given string by replacing values as configured
        :param str string: string to perform replacements on
        :rtype: str
        """
        # For each match, look up the new string in the replacements via the normalized old string
        return self.pattern.sub(lambda match: match.group(1) + self.replacements[self.normalize(match.group(2))] + match.group(3), string)

    def __init__(self, replacements, ignore_case=False):
        """
        Given a replacement map, instantiate a StringReplacer.
        :param dict replacements: replacement dictionary {value to find: value to replace}
        :param bool ignore_case: whether the match should be case insensitive
        :rtype: None
        """
        self.normalize = self._configure_normalize(ignore_case)
        self.replacements = self._configure_replacements(replacements, self.normalize)
        self.pattern = self._configure_pattern(self.replacements, ignore_case)

    def _configure_normalize(self, ignore_case):
        # If case insensitive, we need to normalize the old string so that later a replacement
        # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
        # "HEY", "hEy", etc.
        return lowercase if ignore_case else identity

    def _configure_replacements(self, replacements, normalize):
        return {normalize(key): value for key, value in replacements.items()}

    def _configure_pattern(self, replacements, ignore_case):
        # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
        # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
        # 'hey ABC' and not 'hey ABc'
        sorted_replacement_keys = sorted(replacements, key=len, reverse=True)
        # escaped_replacement_keys = [f"(^|\W)({x})(\W|$)" for x in sorted_replacement_keys]
        escaped_replacement_keys = [re.escape(key) for key in sorted_replacement_keys]

        re_mode = re.IGNORECASE if ignore_case else 0

        # Create a big OR regex that matches any of the substrings to replace
        # add additional regex to only match when at the beginning/end of string or a non-alphanumeric token
        return re.compile("(^|\W)(" + '|'.join(escaped_replacement_keys) + ")(\W|$)", re_mode)


class StringReplacerLite:

    def __init__(self, d):
        self.replacements = {f"_{i}": v for i, (_, v) in enumerate(sorted(d.items()))}
        self.pattern = re.compile("|".join([f"(?P<_{i}>" + k + ")" for i, (k, _) in enumerate(sorted(d.items()))]))

    def __call__(self, string):
        return self.pattern.sub(lambda match: self.replacements[[x for x, v in match.groupdict().items() if v is not None][0]], string)


if __name__ == "__main__":
    from data_normalizers import ERROR_REPLACER
    print(ERROR_REPLACER("Örn är örnets största örn."))
