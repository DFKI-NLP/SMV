from tkinter import ttk
import tkinter as tk
from tkinter import simpledialog
import json


class MainMenu(tk.Frame):
    st_height = 2
    st_width = 20
    colors = [
        "red",
        "dark orange",
        "gold",
        "yellow",
        "light green",
        "green",
        "dark green"
    ]
    feedback = {}
    data_path = ""
    data = None
    valid_keys = []
    currentSID = -1

    def __init__(self):
        self.root = tk.Tk()
        super().__init__(self.root)
        self.grid()
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.ExplainMode = tk.Button(self, fg="red", text="Explain mode", command=self.foo,
                                     width=self.st_width, height=self.st_height)

        self.ReviewMode = tk.Button(self, fg="black", text="Review mode", command=self.review,
                                    width=self.st_width, height=self.st_height)

        self.Settings = tk.Button(self, fg="red", text="Settings", command=self.foo,
                                  width=self.st_width, height=self.st_height)

        self.Quit = tk.Button(self, fg="black", text="Quit", command=self._quit,
                              width=self.st_width, height=self.st_height)

        self.Labels = [tk.Label(self, fg="black", height=self.st_height, width=self.st_width) for i in range(10)]

        self.Buttons = [tk.Button(self, fg="black", height=self.st_height, width=self.st_width) for i in range(20)]

        self.Texts = [tk.Text(self, fg="black", height=self.st_height, width=self.st_width) for i in range(5)]

        self.widgets = [self.ExplainMode, self.ReviewMode, self.Settings, self.Quit]
        self.mainmenu()

    def mainmenu(self):
        self.hideall()
        self.root.geometry("300x160")
        self.root.title("Explainer main menu")
        self.ExplainMode.grid(row=0, column=0)

        self.Labels[0]["text"] = "!Not Implemented!"
        self.Labels[0]["fg"] = "red"
        self.Labels[0].grid(row=0, column=1)

        self.ReviewMode.grid(row=1, column=0)

        self.Settings.grid(row=2, column=0)

        self.Labels[1]["text"] = "!Not Implemented!"
        self.Labels[1]["fg"] = "red"
        self.Labels[1].grid(row=2, column=1)

        self.Quit.grid(row=3, column=0)

        self.update()

    def review(self):
        self.hideall()
        self.root.geometry("1680x720")
        self.root.title("Reviewer")

        self.Texts[0].insert("1.0", "sample text\n"*25 if not self.data else self.data[self.valid_keys[self.currentSID]]["sample"])
        self.Texts[0]["fg"] = "blue"
        self.Texts[0]["height"] = self.st_height*10
        self.Texts[0]["width"] = self.st_width*5
        self.Texts[0].grid(row=0, column=0)

        self.Texts[1].insert("1.0", "sample explanation\n"*25 if not self.data else self.data[self.valid_keys[self.currentSID]]["verbalization"])
        self.Texts[1]["fg"] = "blue"
        self.Texts[1]["height"] = self.st_height*10
        self.Texts[1]["width"] = self.st_width*4
        self.Texts[1].grid(row=0, column=1)

        self.Buttons[0]["command"] = self.show_hm
        self.Buttons[0]["text"] = "show heatmap"
        self.Buttons[0]["fg"] = "black"
        self.Buttons[0]["width"] = self.st_width
        self.Buttons[0]["height"] = self.st_height
        self.Buttons[0].grid(row=0, column=2)

        self.Buttons[1]["command"] = self.rate
        self.Buttons[1]["text"] = "rate sample explanation"
        self.Buttons[1]["fg"] = "black"
        self.Buttons[1]["width"] = self.st_width
        self.Buttons[1]["height"] = self.st_height
        self.Buttons[1].grid(row=1, column=2)

        self.Buttons[2]["command"] = self.mainmenu
        self.Buttons[2]["text"] = "Main menu"
        self.Buttons[2]["fg"] = "black"
        self.Buttons[2]["width"] = self.st_width
        self.Buttons[2]["height"] = self.st_height
        self.Buttons[2].grid(row=2, column=2)

        self.Settings.grid(row=3, column=2)

        self.Buttons[3]["command"] = self.next_s
        self.Buttons[3]["text"] = "Next"
        self.Buttons[3]["fg"] = "black"
        self.Buttons[3]["width"] = self.st_width
        self.Buttons[3]["height"] = self.st_height
        self.Buttons[3].grid(row=4, column=2)

        self.Buttons[4]["command"] = self.loads
        self.Buttons[4]["text"] = "Load json"
        self.Buttons[4]["fg"] = "black"
        self.Buttons[4]["width"] = self.st_width
        self.Buttons[4]["height"] = self.st_height
        self.Buttons[4].grid(row=5, column=2)

        self.Quit.grid(row=6, column=2)

    def foo(self):
        pass

    def next_s(self):
        if self.data:
            self.hideall()
            self.Texts[0].delete("1.0", tk.END)
            self.Texts[1].delete("1.0", tk.END)
            if self.currentSID < len(self.valid_keys) - 1:
                self.currentSID += 1
            else:
                self.currentSID = 0
            self.review()

    def set_sample_sentiment_guess(self, guess):
        self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"] = guess
        self.Labels[1]["text"] = "Your choice: {}".format(
            "positive" if self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"] > 0 else "negative"
        )
        self.Labels[1]["bg"] = "green" if self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"] > 0 else "red"
        self.Labels[1]["fg"] = "black"
        self.Labels[1]["width"] = self.st_width
        self.Labels[1]["height"] = self.st_height
        self.Labels[1].grid(row=1, column=3)
        self.update()

    def set_understandability(self, choice):
        self.feedback[self.valid_keys[self.currentSID]]["Understandability"] = choice
        self.Labels[3]["text"] = "Your choice: {}".format(self.feedback[self.valid_keys[self.currentSID]]["Understandability"])
        self.Labels[3]["bg"] = self.colors[self.feedback[self.valid_keys[self.currentSID]]["Understandability"] - 1]
        self.Labels[3]["fg"] = "black"
        self.Labels[3]["width"] = self.st_width
        self.Labels[3]["height"] = self.st_height
        self.Labels[3].grid(row=3, column=8)
        self.update()

    def set_helpfulness(self, choice):
        self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"] = choice
        self.Labels[5]["text"] = "Your choice: {}".format(self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"])
        self.Labels[5]["bg"] = self.colors[self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"] - 1]
        self.Labels[5]["fg"] = "black"
        self.Labels[5]["width"] = self.st_width
        self.Labels[5]["height"] = self.st_height
        self.Labels[5].grid(row=5, column=8)
        self.update()

    def hideall(self):
        for i in self.widgets:
            i.grid_forget()
            i["bg"] = '#f0f0f0'
        for i in self.Labels:
            i.grid_forget()
            i["bg"] = '#f0f0f0'
        for i in self.Buttons:
            i.grid_forget()
            i["bg"] = '#f0f0f0'
        for i in self.Texts:
            i.grid_forget()
            i["bg"] = '#f0f0f0'
        self.update()

    def show_hm(self):
        pass

    def loads(self):
        USER_INP = simpledialog.askstring(title="Loading",
                                          prompt="Enter Filepath")
        with open(USER_INP, mode="r") as f:
            self.data = json.load(f)

        for i in self.data.keys():
            if i != "modelname":
                self.valid_keys.append(i)

        print("File loaded")

    def rate(self):
        try:
            self.feedback[self.valid_keys[self.currentSID]]
        except KeyError:
            self.feedback[self.valid_keys[self.currentSID]] = {}

        self.hideall()
        self.root.geometry("1600x720")

        self.Labels[0]["text"] = "What did the model predict?"
        self.Labels[0]["fg"] = "black"
        self.Labels[0]["bg"] = "yellow"
        self.Labels[0]["height"] = self.st_height
        self.Labels[0]["width"] = self.st_width * 2
        self.Labels[0]["justify"] = tk.LEFT
        self.Labels[0].grid(row=0, column=0)

        self.Buttons[0]["text"] = "Postitive sentiment"
        self.Buttons[0]["fg"] = "black"
        self.Buttons[0]["bg"] = "green"
        self.Buttons[0]["width"] = self.st_width
        self.Buttons[0]["height"] = self.st_height
        self.Buttons[0]["command"] = lambda: self.set_sample_sentiment_guess(1)
        self.Buttons[0].grid(row=1, column=2)

        self.Buttons[1]["text"] = "Negative sentiment"
        self.Buttons[1]["fg"] = "black"
        self.Buttons[1]["bg"] = "red"
        self.Buttons[1]["width"] = self.st_width
        self.Buttons[1]["height"] = self.st_height
        self.Buttons[1]["command"] = lambda: self.set_sample_sentiment_guess(-1)
        self.Buttons[1].grid(row=1, column=1)

        self.Labels[1]["text"] = ""

        self.Labels[2]["text"] = "How understandable was this explanation?"
        self.Labels[2]["fg"] = "black"
        self.Labels[2]["bg"] = "yellow"
        self.Labels[2]["height"] = self.st_height
        self.Labels[2]["width"] = self.st_width * 2
        self.Labels[2]["justify"] = tk.LEFT
        self.Labels[2].grid(row=2, column=0)

        for i in range(7):
            self.Buttons[2 + i]["text"] = str(i + 1)
            self.Buttons[2 + i]["fg"] = "black"
            self.Buttons[2 + i]["bg"] = self.colors[i]
            self.Buttons[2 + i]["width"] = self.st_width
            self.Buttons[2 + i]["height"] = self.st_height
            self.Buttons[2 + i]["command"] = lambda i=i: self.set_understandability(i+1)
            self.Buttons[2 + i].grid(row=3, column=1+i, sticky=tk.E + tk.W)

        self.Labels[3]["text"] = ""

        self.Labels[4]["text"] = "How understandable was this explanation?"
        self.Labels[4]["fg"] = "black"
        self.Labels[4]["bg"] = "yellow"
        self.Labels[4]["height"] = self.st_height
        self.Labels[4]["width"] = self.st_width * 2
        self.Labels[4]["justify"] = tk.LEFT
        self.Labels[4].grid(row=4, column=0)

        for i in range(7):
            self.Buttons[9 + i]["text"] = str(i + 1)
            self.Buttons[9 + i]["fg"] = "black"
            self.Buttons[9 + i]["bg"] = self.colors[i]
            self.Buttons[9 + i]["width"] = self.st_width
            self.Buttons[9 + i]["height"] = self.st_height
            self.Buttons[9 + i]["command"] = lambda i=i: self.set_helpfulness(i+1)
            self.Buttons[9 + i].grid(row=5, column=1+i, sticky=tk.E + tk.W)

        self.Labels[5]["text"] = ""
        self.Labels[6]["text"] = ""
        self.Labels[6].grid(row=6, column=0)

        self.Quit.grid(row=7, column=1)

        self.Buttons[17]["command"] = self.review
        self.Buttons[17]["text"] = "Back"
        self.Buttons[17]["bg"] = "gray"
        self.Buttons[17]["fg"] = "black"
        self.Buttons[17].grid(row=7, column=2)

        if self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"]:
            self.Labels[1]["text"] = "Your choice: {}".format(
                "positive" if self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"] > 0 else "negative"
            )
            self.Labels[1]["bg"] = "green" if self.feedback[self.valid_keys[self.currentSID]]["SentimentGuess"] > 0 else "red"
            self.Labels[1]["fg"] = "black"
            self.Labels[1]["width"] = self.st_width
            self.Labels[1]["height"] = self.st_height
            self.Labels[1].grid(row=1, column=3)

        if self.feedback[self.valid_keys[self.currentSID]]["Understandability"]:
            self.Labels[3]["text"] = "Your choice: {}".format(
                self.feedback[self.valid_keys[self.currentSID]]["Understandability"])
            self.Labels[3]["bg"] = self.colors[self.feedback[self.valid_keys[self.currentSID]]["Understandability"] - 1]
            self.Labels[3]["fg"] = "black"
            self.Labels[3]["width"] = self.st_width
            self.Labels[3]["height"] = self.st_height
            self.Labels[3].grid(row=3, column=8)

        if self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"]:
            self.Labels[5]["text"] = "Your choice: {}".format(
                self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"])
            self.Labels[5]["bg"] = self.colors[self.feedback[self.valid_keys[self.currentSID]]["Helpfulness"] - 1]
            self.Labels[5]["fg"] = "black"
            self.Labels[5]["width"] = self.st_width
            self.Labels[5]["height"] = self.st_height
            self.Labels[5].grid(row=5, column=8)
        self.update()

    def _quit(self):
        if self.data:
            f = open("review.json", mode="w")
            _ = json.dumps(self.feedback)
            f.write(_)
            f.close()
        self.quit()

    def show(self):
        try:
            f = open("review.json", mode="r")
            self.feedback = json.load(f)
            print(self.feedback)
            f.close()
        except FileNotFoundError:
            pass
        self.mainloop()


