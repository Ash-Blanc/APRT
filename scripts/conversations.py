
import dataclasses
from enum import auto, Enum
from typing import List, Dict


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA3 = auto()
    LLAMA2_chat = auto()
    vicuna = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            seps = [self.sep, self.sep2]
            ret = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += "<|start_header_id|>" + role + "<|end_header_id|>\n\n"  + message + "<|eot_id|>"
                else:
                    ret += "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
                
            return ret
        elif self.sep_style == SeparatorStyle.vicuna:
            #seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    ret += "USER: " + message + " ASSISTANT:"
                else:
                    ret += message + " </s>"
                #if message:
                #    ret += "<|start_header_id|>" + role + "<|end_header_id|>\n\n"  + message + "<|eot_id|>"
                #else:
                #    ret += "<|start_header_id|>" + role + "<|end_header_id|>\n\n"

            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2_chat:
            ret = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""
            ret = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\n"
            '''
            <s> [INST] <<SYS>>\nYou are a pirate chatbot who always responds in pirate speak!\n<</SYS>>\n\nWho are you? [/INST]
            '''
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += message + " [/INST] "
                    else:
                        ret += message + " </s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_source_text(self, message: str):
        """get the source text for mask label"""
        return self.sep + self.roles[0] + ": " + message + self.sep + self.roles[1] + ":"

    def get_target_text(self, message: str):
        """get the source text for mask label"""
        return message + self.sep

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """Register a new conversation template."""
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="train",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="Llama3",
        roles=("user", "assistant"),
        system="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA3,
        sep=" ",
        sep2="</s>",
        stop_token_ids=[2],
    )
)
register_conv_template(
    Conversation(
        name="Llama2_chat",
        roles=("user", "assistant"),
        system="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2_chat,
        sep=" ",
        sep2="[/INST]",
        stop_token_ids=[2],
    )
)

register_conv_template(
    Conversation(
        name="vicuna",
        roles=("user", "assistant"),
        system="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.vicuna,
        sep=" ",
        sep2="</s>",
        stop_token_ids=[2],
    )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna")
    conv.append_message(conv.roles[0], "hello1")
    conv.append_message(conv.roles[1], "hello2")
    print(conv.get_prompt())
