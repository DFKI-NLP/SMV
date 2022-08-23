from tkinter import ttk
import tkinter as tk


class MainMenu(tk.Frame):
    st_height = 2
    st_width = 20

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

        self.Quit = tk.Button(self, fg="black", text="Quit", command=self.quit,
                              width=self.st_width, height=self.st_height)

        self.Labels = [tk.Label(self, fg="black", height=self.st_height, width=self.st_width) for i in range(5)]

        self.Buttons = [tk.Button(self, fg="black", height=self.st_height, width=self.st_width) for i in range(10)]

        self.Texts = [tk.Text(self, fg="black", height=self.st_height, width=self.st_width) for i in range(5)]

        self.widgets = [self.ExplainMode, self.ReviewMode, self.Settings, self.Quit]
        self.mainmenu()

    def mainmenu(self):
        self.root.geometry("300x160")
        self.root.title("Explainer main menu")
        self.hideall()
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
        self.root.geometry("1280x720")
        self.root.title("Reviewer")

        self.Texts[0].insert("1.0", "sample text\n"*25)
        self.Texts[0]["fg"] = "blue"
        self.Texts[0]["height"] = self.st_height*10
        self.Texts[0]["width"] = self.st_width*4
        self.Texts[0].grid(row=0, column=0)

        self.Texts[1].insert("1.0", "sample explanation\n"*25)
        self.Texts[1]["fg"] = "blue"
        self.Texts[1]["height"] = self.st_height*10
        self.Texts[1]["width"] = self.st_width*3
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

        self.Quit.grid(row=4, column=2)




    def foo(self):
        pass

    def hideall(self):
        for i in self.widgets:
            i.grid_forget()
        for i in self.Labels:
            i.grid_forget()
        for i in self.Buttons:
            i.grid_forget()
        for i in self.Texts:
            i.grid_forget()

        self.update()

    def show_hm(self):
        pass

    def rate(self):
        pass

    def show(self):
        self.mainloop()


