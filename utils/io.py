import os
import sys
import random
import datetime
import functools
import contextlib
from typing import Union, TextIO

def root():
    for p in sys.path:
        if p.endswith('AwesomePruning'):
            return p
    else:
        raise ValueError('Cannot find root path of the project')
    
def generate_name() -> str:
    """Random generate a name.

    Returns:
        str: name
    """
    adjectives = [
        "admiring",
        "adoring",
        "affectionate",
        "agitated",
        "amazing",
        "angry",
        "awesome",
        "blissful",
        "bold",
        "boring",
        "brave",
        "busy",
        "charming",
        "clever",
        "cool",
        "compassionate",
        "competent",
        "confident",
        "crazy",
        "dazzling",
        "determined",
        "distracted",
        "dreamy",
        "eager",
        "ecstatic",
        "elastic",
        "elated",
        "elegant",
        "eloquent",
        "epic",
        "exciting",
        "fervent",
        "festive",
        "flamboyant",
        "focused",
        "friendly",
        "frosty",
        "funny",
        "gallant",
        "gifted",
        "goofy",
        "gracious",
        "happy",
        "hardcore",
        "heuristic",
        "hopeful",
        "hungry",
        "infallible",
        "inspiring",
        "jolly",
        "jovial",
        "keen",
        "kind",
        "laughing",
        "loving",
        "lucid",
        "mystifying",
        "modest",
        "musing",
        "naughty",
        "nervous",
        "nice",
        "nifty",
        "nostalgic",
        "objective",
        "optimistic",
        "peaceful",
        "pedantic",
        "pensive",
        "practical",
        "priceless",
        "quirky",
        "quizzical",
        "relaxed",
        "reverent",
        "romantic",
        "sad",
        "serene",
        "sharp",
        "silly",
        "sleepy",
        "stoic",
        "stupefied",
        "suspicious",
        "tender",
        "thirsty",
        "trusting",
        "unruffled",
        "upbeat",
        "vibrant",
        "vigilant",
        "vigorous",
        "wizardly",
        "wonderful",
        "xenodochial",
        "youthful",
        "zealous",
        "zen",
    ]
    scientists = [
        "albattani",
        "allen",
        "almeida",
        "agnesi",
        "archimedes",
        "ardinghelli",
        "aryabhata",
        "austin",
        "babbage",
        "banach",
        "bardeen",
        "bartik",
        "bassi",
        "beaver",
        "bell",
        "benz",
        "bhabha",
        "bhaskara",
        "blackwell",
        "bohr",
        "booth",
        "borg",
        "bose",
        "boyd",
        "brahmagupta",
        "brattain",
        "brown",
        "carson",
        "chandrasekhar",
        "chatelet",
        "chatterjee",
        "chebyshev",
        "cohen",
        "chaum",
        "clarke",
        "colden",
        "cori",
        "cray",
        "curran",
        "curie",
        "darwin",
        "davinci",
        "dewdney",
        "dhawan",
        "diffie",
        "dijkstra",
        "dirac",
        "driscoll",
        "dubinsky",
        "easley",
        "edison",
        "einstein",
        "elbakyan",
        "elgamal",
        "elion",
        "ellis",
        "engelbart",
        "euclid",
        "euler",
        "faraday",
        "feistel",
        "fermat",
        "fermi",
        "feynman",
        "franklin",
        "gagarin",
        "galileo",
        "galois",
        "ganguly",
        "gates",
        "gauss",
        "germain",
        "golick",
        "goodall",
        "gould",
        "greider",
        "grothendieck",
        "haibt",
        "hamilton",
        "hasse",
        "hawking",
        "hellman",
        "heisenberg",
        "hermann",
        "herschel",
        "hertz",
        "heyrovsky",
        "hodgkin",
        "hofstadter",
        "hoover",
        "hopper",
        "hugle",
        "hypatia",
        "ishizaka",
        "jackson",
        "jang",
        "jennings",
        "jepsen",
        "johnson",
        "joliot",
        "jones",
        "kalam",
        "kapitsa",
        "kare",
        "keldysh",
        "keller",
        "kepler",
        "khayyam",
        "khorana",
        "kilby",
        "kirch",
        "knuth",
        "kowalevski",
        "lalande",
        "lamarr",
        "lamport",
        "leakey",
        "leavitt",
        "lewin",
        "lichterman",
        "liskov",
        "lovelace",
        "lumiere",
        "mahavira",
        "margulis",
        "matsumoto",
        "maxwell",
        "mayer",
        "mccarthy",
        "mcclintock",
        "mclean",
        "mcnulty",
        "mendel",
        "mendeleev",
        "meitner",
        "meninsky",
        "merkle",
        "mestorf",
        "mirzakhani",
        "moore",
        "morse",
        "murdock",
        "neumann",
        "newton",
        "nightingale",
        "nobel",
        "nocard",
        "northcutt",
        "noether",
        "norton",
        "noyce",
        "panini",
        "pare",
        "pascal",
        "pasteur",
        "payne",
        "perlman",
        "pike",
        "poincare",
        "poitras",
        "proskuriakova",
        "ptolemy",
        "raman",
        "ramanujan",
        "ride",
        "montalcini",
        "ritchie",
        "robinson",
        "roentgen",
        "rosalind",
        "rubin",
        "saha",
        "sammet",
        "sanderson",
        "satoshi",
        "shamir",
        "shannon",
        "shaw",
        "shirley",
        "shockley",
        "shtern",
        "sinoussi",
        "snyder",
        "solomon",
        "spence",
        "stonebraker",
        "sutherland",
        "swanson",
        "swartzlander",
        "swirles",
        "taussig",
        "tereshkova",
        "tesla",
        "tharp",
        "thompson",
        "torvalds",
        "tu",
        "turing",
        "varahamihira",
        "vaughan",
        "visvesvaraya",
        "volhard",
        "wescoff",
        "wiles",
        "williams",
        "wilson",
        "wing",
        "wozniak",
        "wright",
        "yalow",
        "yonath",
        "zhukovsky",
    ]
    return f"{random.choice(adjectives)}_{random.choice(scientists)}_{datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}"

def soft_mkdir(p:str) -> None:
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(p):
        os.makedirs(p)
        return True
    return False

def log(msg:str, level:str="INFO", file=None) -> None:
    """
    Format and print logs, with optional file output.

    Args:
        msg (str): log message
        level (str, optional): log level. Defaults to "INFO".
        file (Union[str, TextIO, None], optional): 
            - None: print to console only
            - str: if it is a path string, write to the log.txt file in the given path
            - file object: write to the provided file object directly
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{current_time}] [{level}]\t{msg}"
    
    # print to console
    print(log_message)
    
    # handle file writing
    if file is not None:
        try:
            if isinstance(file, str):
                # If it's a path string, write to log.txt in that path
                if not os.path.exists(file):
                    os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
                log_file_path = os.path.join(file, "log.txt") if os.path.isdir(file) else file
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            else:
                # Check if the file object is writable
                if hasattr(file, 'writable') and not file.writable():
                    print(f"Warning: The provided file object is not writable. Log message not written to file.")
                    return
                
                # Assume it's a file object, write directly
                try:
                    file.write(log_message + "\n")
                    file.flush()  # Ensure immediate writing
                except (AttributeError, IOError) as e:
                    print(f"Warning: Failed to write to the provided file object: {e}")
        except Exception as e:
            # Don't let logging errors affect the main program flow
            print(f"Warning: Error occurred while writing log to file: {e}")

class LogRedirectMixin:
    """Log Redirect Mixin, Redirect stdout to a log file, and save the log file in the given path.
       Usage:
        1. Initialize the class with the parameter "log_path".
        2. Call the log_decorator decorator to wrap the desired method.
    """
    def __init__(self, log_path: str=None):
        if not log_path:
            log_path = os.path.join(root(), 'experiments', generate_name())
            self.log_path = log_path
        flag = soft_mkdir(log_path)
        self.log_file_path = os.path.join(log_path, 'log.txt')
        if not flag:
            print("Log path already exists.")
            name = str(self.__class__).replace("<class '", "").replace("'>", "")
            self.log_file_path = os.path.join(log_path, f'log_{name}.txt')
            # raise Warning("Log path already exists.")
        print(f"[INFO] Log saved in {log_path}")
        self._wrap_methods()
    
    def log_decorator(self, func):
        """Decorator to redirect stdout to the log file during function execution"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(self.log_file_path, 'a') as f:
                with contextlib.redirect_stdout(f):
                    result = func(*args, **kwargs)
                sys.stdout = original_stdout
                result = func(*args, **kwargs)
            return result
        return wrapper

    def _wrap_methods(self):
        """Wrap all user-defined methods with the log decorator"""
        for attribute_name in dir(self):
            if attribute_name.startswith("__") and attribute_name.endswith("__"):
                continue  # Skip special methods
            attribute = getattr(self, attribute_name)
            if callable(attribute):
                # Wrap the method with the log decorator
                setattr(self, attribute_name, self.log_decorator(attribute))