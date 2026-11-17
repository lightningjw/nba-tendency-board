"""Named placeholders for everything outside the MVP. None of these are implemented.

Each stub says where it plugs in so Phase 2/3 work does not need to touch the contract's
existing fields. See NEXT.md.
"""


class NotBuilt(NotImplementedError):
    def __init__(self, what: str, phase: str):
        super().__init__(f"{what} is not part of the MVP ({phase}). See NEXT.md.")
