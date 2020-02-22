try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def plot_no_player_aces(q):
    fig, axis = plt.subplots(3, 6, sharey=True)

    fig.suptitle('Player Hand Values (No Aces) vs Dealer First Card Value', fontsize=10)
    size = 5

    for index, player_start in enumerate(range(4, 22)):
        p = frozenset(set([player_start]))

        hits = list()
        stays = list()
        for dealer_start in range(1, 12):
            d = frozenset(set([dealer_start]))
            if dealer_start == 1 or dealer_start == 11:
                d = frozenset([1, 11])
            if p in q and d in q[p]:
                values = q[p][d]
                hits.append(values[0])
                stays.append(values[1])
            else:
                hits.append(None)
                stays.append(None)

        plt.subplot(3, 6, 1 + index)
        plt.title('player hand value: ' + str(player_start), fontsize=6)
        ax = plt.gca()

        ax.set_xlabel('dealer first card', fontsize=5)
        ax.set_ylabel('reward', fontsize=5)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        plt.plot(range(1, len(hits) + 1), hits, 'g-', label='hits')
        plt.plot(range(1, len(stays) + 1), stays, 'r-', label='stays')

        if player_start == 21:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')

    plt.tight_layout(pad=0.2, h_pad=0.4, w_pad=-1.4)
    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.show()


def print_no_player_aces(q):
    for player_start in range(4, 22):
        for dealer_start in range(1, 12):
            p = frozenset(set([player_start]))
            d = frozenset(set([dealer_start]))
            if dealer_start == 1 or dealer_start == 11:
                d = frozenset([1, 11])

            if p in q and d in q[p]:
                values = q[p][d]
                hit = str(round(values[0], 3)).ljust(6, '0')
                stay = str(round(values[1], 3)).ljust(6, '0')
            else:
                hit = "-"
                stay = "-"

            print('{} / {}: [{}, {}]'.format(str(player_start).rjust(2, ' '), str(dealer_start).rjust(2, ' '), hit, stay))


def print_one_player_ace(q):
    print('fit, one player ace, dealer ace = 1 or 11:')
    for player_start in range(2, 21):
        for dealer_start in range(1, 12):
            p = frozenset(set([player_start + 1, player_start + 11]))
            d = frozenset(set([dealer_start]))
            if dealer_start == 1 or dealer_start == 11:
                d = frozenset([1, 11])

            if p in q and d in q[p]:
                values = q[p][d]
                hit = str(round(values[0], 3)).ljust(6, '0')
                stay = str(round(values[1], 3)).ljust(6, '0')
            else:
                hit = "-"
                stay = "-"
            print('{} / {}: [{}, {}]'.format(str([player_start + 1, player_start + 11]).rjust(8, ' '), str(dealer_start).rjust(2, ' '), hit, stay))
