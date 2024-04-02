

## daca este mai mare de 0,36 deja nu mai are treba cu ce am intrebat eu
def scriem_in_fiser(intrebare, raspuns):

    with open('./conversatie_text.txt', 'w', encoding='utf-8' ) as file:
        if intrebare: file.write('Intrebare:  ' + intrebare)

        for prop in raspuns:
            file.write('\n\n' + 'Raspuns:  ' + prop)

#######################################################################
def scriem_in_fiserMesAi(arrayCuObiect):
    with open('./conversatie_text_MesAi.txt', 'w', encoding='utf-8') as file:

        for obiect in arrayCuObiect:
            file.write('\n\n' + str(obiect))


######################################################################
## => returneaza numarul de cuvinte dintr-o  fraza
def numarDeCuvinte(frazaString):
    arCuCuv = frazaString.split()
    return len(arCuCuv)

# print(numarDeCuvinte('''
# A ROADMAP TO YOUR JOURNEY TO FINANCIAL SECURITY   |  19WARNING!Before Y ou Invest Always Check with the SEC and Y our State’s
# Securities Regulator:
# Is the investment registered?Have investors complained about the investment in the past?Have the people who own or manage the investment been in trouble in the past?Is the person selling me this investment licensed in my state?Has that person been in trouble with the SEC, my state, or other investors in the past?
# '''))
