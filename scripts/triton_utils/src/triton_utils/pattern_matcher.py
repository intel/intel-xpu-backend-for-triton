import re
from dataclasses import dataclass, field


@dataclass
class PatternMatcher:
    include_patterns: list[re.Pattern[str]] = field(default_factory=list, )
    exclude_patterns: list[re.Pattern[str]] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        if self.exclude_patterns:
            for pattern in self.exclude_patterns:
                if pattern.search(text):
                    return False

        if self.include_patterns:
            for pattern in self.include_patterns:
                if pattern.search(text):
                    return True
            return False

        return True
