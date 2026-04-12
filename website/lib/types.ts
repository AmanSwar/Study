export interface Track {
  id: string
  title: string
  shortTitle: string
  description: string
  moduleCount: number
  appendixCount: number
  icon: string
  color: 'blue' | 'cyan' | 'orange' | 'green'
  parts?: Part[]
  modules?: ModuleMeta[] // for flat tracks like Qualcomm
}

export interface Part {
  id: string
  number: number
  title: string
  shortTitle: string
  description: string
  modules: ModuleMeta[]
}

export interface ModuleMeta {
  id: string
  number: number
  title: string
  shortTitle: string
  description: string
  readingTime: string
  sourceFile: string
  href: string
}

export interface AppendixMeta {
  id: string
  letter: string
  title: string
  href: string
}
